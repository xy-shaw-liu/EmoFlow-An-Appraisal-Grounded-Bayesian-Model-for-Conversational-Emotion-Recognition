"""
EmoFlow DataLoader.

Loads preprocessed Dialogue JSONL ([[preprocess.py]] output), applies label
normalization to a unified emotion vocabulary, attaches Scherer Table 5.5
appraisal targets ([[appraisal_targets.py]]) with per-utterance masks, and
batches dialogues with padding.

Decisions encoded here (project memory: emoflow-data-pipeline,
emoflow-appraisal-targets):
- LABEL_NORMALIZE applied to meld + dailydialog (joy/happiness → joy etc.)
- EmoryNLP keeps all 7 native labels (joyful/sad/mad/scared/peaceful/
  powerful/neutral) for the classification head; for the appraisal head,
  only joyful/sad/mad/scared map to a Scherer target — peaceful/powerful/
  neutral are masked out of MSE loss.
- DailyDialog appraisal pretraining: pass `filter_no_emotion_dialogues=True`
  to drop dialogues where every utterance is no_emotion; weight MSE by
  `class_weights()` computed on the filtered set.
- Speaker is mapped to a dialogue-local integer (first appearance = 0).

Output of `collate_dialogues`:
  text             : list[list[str]]  — (B, T_max) raw text, no tokenization
  speaker_idx      : np.int64    (B, T_max)
  turn_idx         : np.int64    (B, T_max)
  emotion_label_idx: np.int64    (B, T_max)
  appraisal_target : np.float32  (B, T_max, 5)
  appraisal_mask   : np.bool_    (B, T_max)  — True iff utterance has a target
  utt_mask         : np.bool_    (B, T_max)  — True iff slot is a real utterance
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from appraisal_targets import (
    APPRAISAL_DIMS,
    APPRAISAL_TARGETS,
    LABEL_NORMALIZE,
    MASKED_LABELS,
)

DATA_ROOT = Path(__file__).parent / "data_processed"

# Dataset-native label vocabularies (stable index order).
# meld + dailydialog are normalized to the same 7-class space.
UNIFIED_VOCAB = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
EMORYNLP_VOCAB = ["neutral", "joyful", "sad", "mad", "scared",
                  "peaceful", "powerful"]

VOCAB = {
    "meld":        UNIFIED_VOCAB,
    "dailydialog": UNIFIED_VOCAB,
    "emorynlp":    EMORYNLP_VOCAB,
}

# For emorynlp, normalize joyful/sad/mad/scared to canonical Scherer keys for
# appraisal lookup; peaceful/powerful/neutral pass through and get masked.
_EMO_TO_APPRAISAL_KEY = {
    "joyful": "joy", "sad": "sadness", "mad": "anger", "scared": "fear",
}


APPRAISAL_DIM = len(APPRAISAL_DIMS)


@dataclass
class Utterance:
    text: str
    speaker_idx: int
    turn_idx: int
    emotion_label_idx: int
    appraisal_target: np.ndarray   # (APPRAISAL_DIM,) float32, zeros if masked
    appraisal_mask: bool


@dataclass
class Dialogue:
    dialogue_id: str
    dataset: str
    split: str
    utterances: list[Utterance]

    def __len__(self) -> int:
        return len(self.utterances)


def _appraisal_for(label_native: str, dataset: str) -> tuple[np.ndarray, bool]:
    """Return (5-dim target, mask). Mask=False for labels with no Scherer profile."""
    if dataset == "emorynlp":
        key = _EMO_TO_APPRAISAL_KEY.get(label_native)
        if key is None:           # peaceful / powerful / neutral
            return np.zeros(APPRAISAL_DIM, dtype=np.float32), False
    else:
        key = LABEL_NORMALIZE.get(label_native, label_native)
        if key in MASKED_LABELS:  # neutral
            return np.zeros(APPRAISAL_DIM, dtype=np.float32), False
    target = APPRAISAL_TARGETS.get(key)
    if target is None:
        return np.zeros(APPRAISAL_DIM, dtype=np.float32), False
    return target.astype(np.float32), True


class EmoFlowDataset:
    """One Dialogue per index. Dataset-native label vocab via .vocab."""

    def __init__(
        self,
        dataset: str,
        split: str,
        data_root: Path | str = DATA_ROOT,
        filter_no_emotion_dialogues: bool = False,
    ):
        if dataset not in VOCAB:
            raise ValueError(f"unknown dataset: {dataset}")
        self.dataset = dataset
        self.split = split
        self.vocab = VOCAB[dataset]
        self._label_to_idx = {l: i for i, l in enumerate(self.vocab)}

        path = Path(data_root) / dataset / f"{split}.jsonl"
        with path.open() as f:
            raw = [json.loads(line) for line in f]

        if filter_no_emotion_dialogues:
            # drop dialogues where every utterance is the "no-emotion" label
            no_emo = {"no_emotion", "neutral"}
            raw = [d for d in raw
                   if any(u["emotion"] not in no_emo for u in d["utterances"])]

        self._dialogues: list[Dialogue] = [self._build_dialogue(d) for d in raw]

    def _build_dialogue(self, d: dict) -> Dialogue:
        # local speaker index: first-appearance order within the dialogue
        spk_to_idx: dict[str | None, int] = {}
        utts = []
        for u in d["utterances"]:
            spk = u["speaker"]
            if spk not in spk_to_idx:
                spk_to_idx[spk] = len(spk_to_idx)
            raw_label = u["emotion"]
            # meld + dailydialog get normalized into UNIFIED_VOCAB;
            # emorynlp keeps native labels.
            if self.dataset == "emorynlp":
                label = raw_label
            else:
                label = LABEL_NORMALIZE.get(raw_label, raw_label)
            if label not in self._label_to_idx:
                raise ValueError(
                    f"label {raw_label!r} (normalized: {label!r}) "
                    f"not in vocab for {self.dataset}: {self.vocab}"
                )
            target, mask = _appraisal_for(raw_label, self.dataset)
            utts.append(Utterance(
                text=u["text"],
                speaker_idx=spk_to_idx[spk],
                turn_idx=u["turn_idx"],
                emotion_label_idx=self._label_to_idx[label],
                appraisal_target=target,
                appraisal_mask=mask,
            ))
        return Dialogue(
            dialogue_id=d["dialogue_id"],
            dataset=d["dataset"],
            split=d["split"],
            utterances=utts,
        )

    def __len__(self) -> int:
        return len(self._dialogues)

    def __getitem__(self, idx: int) -> Dialogue:
        return self._dialogues[idx]

    def __iter__(self) -> Iterator[Dialogue]:
        return iter(self._dialogues)

    # ----- statistics -----

    def label_distribution(self) -> dict[str, int]:
        c = Counter()
        for d in self._dialogues:
            for u in d.utterances:
                c[self.vocab[u.emotion_label_idx]] += 1
        return dict(c)

    def class_weights(self, smoothing: float = 1.0) -> np.ndarray:
        """Inverse-frequency weights per emotion class, indexed by vocab.

        weight_i = N_total / (K * (count_i + smoothing))
        Returns float32 array of length len(vocab). Use as per-sample
        scalar multiplier on the MSE (or CE) loss.
        """
        counts = np.zeros(len(self.vocab), dtype=np.float64)
        for d in self._dialogues:
            for u in d.utterances:
                counts[u.emotion_label_idx] += 1
        total = counts.sum()
        K = len(self.vocab)
        return (total / (K * (counts + smoothing))).astype(np.float32)


def collate_dialogues(batch: list[Dialogue]) -> dict:
    """Pad a list of Dialogues to the max length in batch."""
    B = len(batch)
    T = max(len(d) for d in batch)

    text = [["" for _ in range(T)] for _ in range(B)]
    speaker_idx       = np.zeros((B, T), dtype=np.int64)
    turn_idx          = np.zeros((B, T), dtype=np.int64)
    emotion_label_idx = np.zeros((B, T), dtype=np.int64)
    appraisal_target  = np.zeros((B, T, APPRAISAL_DIM), dtype=np.float32)
    appraisal_mask    = np.zeros((B, T),    dtype=bool)
    utt_mask          = np.zeros((B, T),    dtype=bool)

    for bi, d in enumerate(batch):
        for ti, u in enumerate(d.utterances):
            text[bi][ti]                 = u.text
            speaker_idx[bi, ti]          = u.speaker_idx
            turn_idx[bi, ti]             = u.turn_idx
            emotion_label_idx[bi, ti]    = u.emotion_label_idx
            appraisal_target[bi, ti]     = u.appraisal_target
            appraisal_mask[bi, ti]       = u.appraisal_mask
            utt_mask[bi, ti]             = True

    return {
        "text": text,
        "speaker_idx": speaker_idx,
        "turn_idx": turn_idx,
        "emotion_label_idx": emotion_label_idx,
        "appraisal_target": appraisal_target,
        "appraisal_mask": appraisal_mask,
        "utt_mask": utt_mask,
        "dialogue_id": [d.dialogue_id for d in batch],
    }


if __name__ == "__main__":
    print("=== Smoke test ===")
    for name in ["meld", "dailydialog", "emorynlp"]:
        ds = EmoFlowDataset(name, "train")
        dist = ds.label_distribution()
        print(f"\n{name} train: {len(ds)} dialogues, vocab={ds.vocab}")
        print(f"  label_dist: {dist}")
        print(f"  class_wts : {dict(zip(ds.vocab, ds.class_weights().round(3)))}")

    print("\n=== DailyDialog with filter_no_emotion_dialogues=True ===")
    ds = EmoFlowDataset("dailydialog", "train", filter_no_emotion_dialogues=True)
    dist = ds.label_distribution()
    print(f"  remaining dialogues: {len(ds)}  (original 11118)")
    print(f"  label_dist: {dist}")
    print(f"  class_wts (filtered): {dict(zip(ds.vocab, ds.class_weights().round(3)))}")

    print("\n=== Collate test (batch of 3 MELD dialogues) ===")
    ds = EmoFlowDataset("meld", "train")
    batch = collate_dialogues([ds[i] for i in range(3)])
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:18s}: shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k:18s}: len={len(v)}  e.g. {repr(v[0])[:60]}")

    print(f"\n  appraisal_target[0, :4]:\n{batch['appraisal_target'][0, :4]}")
    print(f"  appraisal_mask  [0, :8]: {batch['appraisal_mask'][0, :8]}")
    print(f"  utt_mask        [0]    : {batch['utt_mask'][0]}")
    print(f"  emotion labels  [0]    : "
          f"{[ds.vocab[i] for i in batch['emotion_label_idx'][0][:batch['utt_mask'][0].sum()]]}")
