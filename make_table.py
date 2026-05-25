"""
Aggregate all ckpt/*/metrics.json into a results table (CSV + markdown).

  python3 make_table.py                    # scan ckpt/ → results.csv + results.md
  python3 make_table.py --ckpt_root ckpt/  # explicit
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(root: Path) -> list[dict]:
    rows = []
    for metrics_f in sorted(root.glob("*/metrics.json")):
        run = metrics_f.parent.name
        m = json.loads(metrics_f.read_text())
        args = json.loads((metrics_f.parent / "args.json").read_text())
        rows.append({
            "run":         run,
            "model":       args.get("model"),
            "dataset":     args.get("dataset"),
            "backbone":    args.get("backbone"),
            "init_lambda": args.get("init_lambda"),
            "freeze_lambda": args.get("freeze_lambda"),
            "seed":        args.get("seed"),
            "weighted_f1": m.get("weighted_f1"),
            "eta":         m.get("eta"),
            "n_trans":     m.get("n_transitions"),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default="ckpt")
    p.add_argument("--out_csv", default="results.csv")
    p.add_argument("--out_md", default="results.md")
    args = p.parse_args()

    rows = collect(Path(args.ckpt_root))
    if not rows:
        print(f"no metrics.json found under {args.ckpt_root}/")
        return

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} runs)")

    # Markdown
    cols = ["model", "dataset", "backbone", "init_lambda", "freeze_lambda",
            "seed", "weighted_f1", "eta", "n_trans"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
