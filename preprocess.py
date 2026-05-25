"""
Unify EmoryNLP / MELD / DailyDialog into a common Dialogue JSONL format.

Per-line schema (one Dialogue per line):
{
  "dialogue_id": str,
  "dataset":     "emorynlp" | "meld" | "dailydialog",
  "split":       "train" | "dev" | "test",
  "utterances":  [
    {"turn_idx": int, "text": str, "speaker": str | None, "emotion": str}
  ]
}

Label spaces are kept dataset-native (decision: do not force-map EmoryNLP to Ekman):
  emorynlp     : {joyful, mad, neutral, peaceful, powerful, sad, scared}
  meld         : {neutral, joy, sadness, anger, fear, disgust, surprise}
  dailydialog  : {no_emotion, anger, disgust, fear, happiness, sadness, surprise}

Conventions:
  - EmoryNLP dialogue boundary = scene
  - MELD dialogue boundary     = Dialogue_ID
  - DailyDialog speaker        = alternating "A" / "B"
  - turn_idx                   = 0-based, within each Dialogue
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "datasets"
OUT = ROOT / "data_processed"

DD_EMOTION = {
    0: "no_emotion", 1: "anger",  2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise",
}


def write_jsonl(path: Path, dialogues):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ---------- EmoryNLP ----------
def load_emorynlp(split_file: str, split_name: str):
    with (RAW / "emotion-detection/json" / split_file).open() as f:
        data = json.load(f)
    out = []
    for ep in data["episodes"]:
        for sc in ep["scenes"]:
            utts = []
            for i, u in enumerate(sc["utterances"]):
                speakers = u.get("speakers") or [None]
                utts.append({
                    "turn_idx": i,
                    "text": u["transcript"],
                    "speaker": speakers[0],
                    "emotion": u["emotion"].lower(),
                })
            out.append({
                "dialogue_id": sc["scene_id"],
                "dataset": "emorynlp",
                "split": split_name,
                "utterances": utts,
            })
    return out


# ---------- MELD ----------
def load_meld(csv_file: str, split_name: str):
    rows_by_dlg: dict[int, list] = {}
    with (RAW / "MELD/data/MELD" / csv_file).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_dlg.setdefault(int(row["Dialogue_ID"]), []).append(row)
    out = []
    for dlg_id, rows in sorted(rows_by_dlg.items()):
        rows.sort(key=lambda r: int(r["Utterance_ID"]))
        utts = [{
            "turn_idx": i,
            "text": r["Utterance"],
            "speaker": r["Speaker"],
            "emotion": r["Emotion"].lower(),
        } for i, r in enumerate(rows)]
        out.append({
            "dialogue_id": f"meld_{split_name}_{dlg_id}",
            "dataset": "meld",
            "split": split_name,
            "utterances": utts,
        })
    return out


# ---------- DailyDialog ----------
def load_dailydialog(split_name: str):
    from datasets import load_from_disk
    ds = load_from_disk(str(RAW / "daily_dialog"))
    hf_split = {"train": "train", "dev": "validation", "test": "test"}[split_name]
    out = []
    for idx, ex in enumerate(ds[hf_split]):
        utts = []
        for i, (text, emo) in enumerate(zip(ex["dialog"], ex["emotion"])):
            utts.append({
                "turn_idx": i,
                "text": text.strip(),
                "speaker": "A" if i % 2 == 0 else "B",
                "emotion": DD_EMOTION[emo],
            })
        out.append({
            "dialogue_id": f"dd_{split_name}_{idx}",
            "dataset": "dailydialog",
            "split": split_name,
            "utterances": utts,
        })
    return out


def main():
    jobs = [
        ("emorynlp", "train", lambda: load_emorynlp("emotion-detection-trn.json", "train")),
        ("emorynlp", "dev",   lambda: load_emorynlp("emotion-detection-dev.json", "dev")),
        ("emorynlp", "test",  lambda: load_emorynlp("emotion-detection-tst.json", "test")),
        ("meld",     "train", lambda: load_meld("train_sent_emo.csv", "train")),
        ("meld",     "dev",   lambda: load_meld("dev_sent_emo.csv",   "dev")),
        ("meld",     "test",  lambda: load_meld("test_sent_emo.csv",  "test")),
        ("dailydialog", "train", lambda: load_dailydialog("train")),
        ("dailydialog", "dev",   lambda: load_dailydialog("dev")),
        ("dailydialog", "test",  lambda: load_dailydialog("test")),
    ]
    for dataset, split, fn in jobs:
        dialogues = fn()
        n_utt = sum(len(d["utterances"]) for d in dialogues)
        write_jsonl(OUT / dataset / f"{split}.jsonl", dialogues)
        print(f"{dataset:11s} {split:5s}: {len(dialogues):5d} dialogues, {n_utt:6d} utterances")


if __name__ == "__main__":
    sys.exit(main())
