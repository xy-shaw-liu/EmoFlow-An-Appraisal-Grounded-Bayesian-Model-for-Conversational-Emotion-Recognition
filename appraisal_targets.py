"""
Weak-supervision appraisal targets from Scherer (2001) Table 5.5.

Source: Scherer, K. R. (2001). Appraisal Considered as a Process of
Multilevel Sequential Checking. In Scherer, Schorr, & Johnstone (Eds.),
*Appraisal Processes in Emotion*, Ch. 5, Table 5.5 (pp. 92-120).
Reproduced and modified from Scherer (1997a, pp. 138-139).

Table 5.5 reports empirical Z-score appraisal profiles from a 37-country
ISEAR-style study (n ≈ 3000 respondents). Each cell has a verbal prediction
(from an earlier theory version) plus an empirical Z-score; we use the
Z-scores. ISEAR did not include surprise — we derive a surprise profile
theoretically (high novelty, others at population mean = 0).

Pipeline:
  Table 5.5 Z-scores  -- aggregate 8 ISEAR dims --> 5 appraisal dims
                       -- per-dim min-max across 6 emotions --> [0, 1]
  Project label  -- LABEL_NORMALIZE --> unified key
                 -- APPRAISAL_TARGETS[key] --> 5-dim vector  or  mask
"""

import numpy as np

# ----- 1. Table 5.5 raw Z-scores -----
# Columns: emotions. Rows: ISEAR dimensions.
# (shame / guilt are in the table but not in our datasets, so dropped.)
_EMOTIONS = ["joy", "fear", "anger", "sadness", "disgust"]
_Z = {
    # dim:                joy    fear   anger  sadness disgust
    "expectedness":      [ 0.64, -0.12, -0.21,  0.03,  -0.19],
    "unpleasantness":    [-2.00,  0.34,  0.41,  0.36,   0.40],
    "goal_hindrance":    [-1.18,  0.12,  0.37,  0.30,   0.13],
    "external_causation":[-0.13,  0.23,  0.12,  0.47,   0.27],
    "coping_potential":  [ 0.44, -0.29,  0.07, -0.35,   0.02],
    "unfairness":        [-0.72,  0.00,  0.58,  0.11,   0.27],
    "immorality":        [-0.63, -0.05,  0.29, -0.12,   0.29],
    "self_consistency":  [ 1.18, -0.06, -0.09, -0.17,  -0.02],
}

def _col(dim: str, emo: str) -> float:
    return _Z[dim][_EMOTIONS.index(emo)]


# ----- 2. derived surprise -----
# ISEAR omits surprise. Per Scherer's description of "astonishment":
# very low expectedness (= high novelty); all other dimensions ~ population
# mean (Z = 0). Set expectedness to a value below the empirical minimum so
# that surprise becomes max-novelty after min-max scaling.
_SURPRISE_Z = {
    "expectedness":      -1.0,   # below empirical min (-0.21)
    "unpleasantness":     0.0,
    "goal_hindrance":     0.0,
    "external_causation": 0.0,
    "coping_potential":   0.0,
    "unfairness":         0.0,
    "immorality":         0.0,
    "self_consistency":   0.0,
}


# ----- 3. raw 8-dim ISEAR appraisal vector per emotion -----
# We keep the full 8 Scherer/ISEAR dimensions (no aggregation), preserving
# each component's distinct semantic role. Earlier we aggregated to 5 dims;
# the 5-dim bottleneck proved too narrow for emotion discrimination on
# unbalanced dialogue data, so we now expose all 8 raw dims.
APPRAISAL_DIMS = list(_Z.keys())   # 8: expectedness ... self_consistency


def _raw_vec_for(zget) -> np.ndarray:
    return np.array([zget(d) for d in APPRAISAL_DIMS], dtype=np.float32)


_RAW = {emo: _raw_vec_for(lambda d, e=emo: _col(d, e)) for emo in _EMOTIONS}
_RAW["surprise"] = _raw_vec_for(lambda d: _SURPRISE_Z[d])


# ----- 4. min-max per dim to [0, 1] across 6 emotions -----
def build_targets() -> dict[str, np.ndarray]:
    mat = np.stack([_RAW[e] for e in _RAW])  # (6, 8)
    mn, mx = mat.min(axis=0), mat.max(axis=0)
    scale = np.where(mx > mn, mx - mn, 1.0)
    return {e: ((_RAW[e] - mn) / scale).astype(np.float32) for e in _RAW}


APPRAISAL_TARGETS = build_targets()


# ----- 5. label normalization (applied in DataLoader) -----
LABEL_NORMALIZE = {
    # MELD (canonical 6-class + neutral)
    "joy": "joy", "sadness": "sadness", "anger": "anger", "fear": "fear",
    "disgust": "disgust", "surprise": "surprise", "neutral": "neutral",
    # DailyDialog
    "happiness": "joy", "no_emotion": "neutral",
    # EmoryNLP: 4 native classes map cleanly; peaceful / powerful / neutral
    # stay as themselves and get masked from appraisal MSE.
    "joyful": "joy", "sad": "sadness", "mad": "anger", "scared": "fear",
    "peaceful": "peaceful", "powerful": "powerful",
}

MASKED_LABELS = {"neutral", "peaceful", "powerful"}


if __name__ == "__main__":
    import json
    print(f"Appraisal targets (min-max [0,1], 6 emotions × {len(APPRAISAL_DIMS)} dims):")
    print(f"  dims: {APPRAISAL_DIMS}\n")
    for k, vec in APPRAISAL_TARGETS.items():
        rounded = {d: round(float(x), 3) for d, x in zip(APPRAISAL_DIMS, vec)}
        print(f"  {k:9s}: {rounded}")
    print(f"\nMasked labels (no appraisal supervision): {sorted(MASKED_LABELS)}")
    print(f"\nLabel normalization map:\n{json.dumps(LABEL_NORMALIZE, indent=2)}")
