"""
EmoFlow evaluation metrics.

Two metrics per SVG plan:
  weighted-F1   per-turn emotion classification, sklearn weighted F1
  ETA           Emotion Transition Accuracy (version B, per DialogueRNN):
                  on turns where the *true* label differs from the previous
                  turn (i.e. a real transition occurred), what fraction did
                  the model predict correctly?
                  = correctly-classified-transition-turns / total-transition-turns

Both metrics are computed per-dataset on a flat list of (dialogue_id,
turn_idx, true_label, pred_label) tuples.

Coherence Score (the 3rd metric in the SVG) is human-rated, not in this file.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, f1_score


def weighted_f1(y_true: Sequence[int], y_pred: Sequence[int],
                labels: Sequence[int] | None = None) -> float:
    return float(f1_score(y_true, y_pred, labels=labels,
                          average="weighted", zero_division=0))


def emotion_transition_accuracy(
    dialogue_id: Sequence,
    turn_idx: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict:
    """ETA version B (DialogueRNN-style).

    For each transition turn t where y_true[t] != y_true[t-1] within the same
    dialogue, count it correct iff y_pred[t] == y_true[t].

    Returns dict with: eta, n_transitions, n_correct, n_turns.
    """
    by_dlg = defaultdict(list)
    for d, t, yt, yp in zip(dialogue_id, turn_idx, y_true, y_pred):
        by_dlg[d].append((int(t), int(yt), int(yp)))

    n_trans = 0
    n_correct = 0
    n_turns = 0
    for d, rows in by_dlg.items():
        rows.sort(key=lambda r: r[0])
        for (_, yt_prev, _), (_, yt, yp) in zip(rows, rows[1:]):
            n_turns += 1
            if yt != yt_prev:
                n_trans += 1
                if yp == yt:
                    n_correct += 1
    eta = (n_correct / n_trans) if n_trans else float("nan")
    return {
        "eta": eta,
        "n_transitions": n_trans,
        "n_correct": n_correct,
        "n_followup_turns": n_turns,
    }


def compute_metrics(
    dialogue_id: Sequence,
    turn_idx: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] | None = None,
    verbose: bool = False,
) -> dict:
    """Bundle weighted-F1 + ETA-B into one report."""
    wf1 = weighted_f1(y_true, y_pred)
    eta = emotion_transition_accuracy(dialogue_id, turn_idx, y_true, y_pred)
    out = {"weighted_f1": wf1, **eta}
    if verbose and label_names is not None:
        print(classification_report(y_true, y_pred,
                                    target_names=list(label_names),
                                    zero_division=0))
    return out


if __name__ == "__main__":
    # mock test — two short dialogues
    dialogue_id = ["d1"]*4 + ["d2"]*5
    turn_idx    = [0, 1, 2, 3,  0, 1, 2, 3, 4]
    # d1: neutral, neutral, joy, joy           (1 real transition at t=2)
    # d2: anger, anger, sadness, sadness, joy  (2 transitions: t=2, t=4)
    y_true = [0, 0, 1, 1,     2, 2, 3, 3, 1]
    # model gets some right, some wrong
    y_pred = [0, 0, 1, 0,     2, 3, 3, 3, 1]
    # transitions where y_true changes:
    #   d1 t=2: 0→1, true=1 pred=1 ✓
    #   d2 t=2: 2→3, true=3 pred=3 ✓
    #   d2 t=4: 3→1, true=1 pred=1 ✓
    # ETA = 3/3 = 1.0
    names = ["neutral", "joy", "anger", "sadness"]
    m = compute_metrics(dialogue_id, turn_idx, y_true, y_pred,
                        label_names=names, verbose=True)
    print(f"\nmetrics: {m}")
    assert m["n_transitions"] == 3, m
    assert m["n_correct"] == 3, m
    assert m["eta"] == 1.0, m
    print("✓ ETA accounting matches hand calculation")
