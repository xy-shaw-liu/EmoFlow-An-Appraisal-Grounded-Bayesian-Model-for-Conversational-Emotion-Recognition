"""
Train StimulusEncoder + AppraisalHead on DailyDialog (weak supervision).

Per project memory:
  - filter dialogues where every utterance is no_emotion (handled by DataLoader)
  - loss = MSE on appraisal_target (masked to utterances with valid targets)
  - per-sample weight = inverse-frequency class weight (computed on filtered set)

Usage:
  python3 train_appraisal.py                              # defaults (distilbert, small)
  python3 train_appraisal.py --backbone meta-llama/Meta-Llama-3-8B --batch_size 4
  python3 train_appraisal.py --max_steps 50               # quick pipeline check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from appraisal_targets import APPRAISAL_DIMS
from dataloader import EmoFlowDataset, collate_dialogues
from encoder import StimulusEncoder


def flatten_batch(batch: dict, device) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Pull valid utterances out of a padded (B, T) batch.

    Returns (texts, target, weight) where:
      texts  : list[str] of length N
      target : (N, 5) float on device
      weight : (N,)    float on device   — per-sample inverse-freq class weight
    """
    utt_mask = batch["utt_mask"] & batch["appraisal_mask"]   # (B, T)
    flat_idx = np.argwhere(utt_mask)                          # (N, 2)
    texts = [batch["text"][b][t] for b, t in flat_idx]
    target = torch.from_numpy(
        batch["appraisal_target"][utt_mask]
    ).to(device)                                              # (N, 5)
    label_idx = batch["emotion_label_idx"][utt_mask]          # (N,)
    weight = torch.from_numpy(
        batch["_class_weights"][label_idx]
    ).to(device)                                              # (N,)
    return texts, target, weight


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted MSE, averaged over batch and dims."""
    per_sample = ((pred - target) ** 2).mean(dim=-1)           # (N,)
    return (per_sample * weight).sum() / weight.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model: StimulusEncoder, loader: DataLoader,
             device, max_batches: int | None = None) -> dict:
    model.eval()
    se = torch.zeros(5, device=device)
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        texts, target, _ = flatten_batch(batch, device)
        if not texts:
            continue
        enc = model.encode_text(texts, device)
        pred = model(**enc)
        se += ((pred - target) ** 2).sum(dim=0)
        n += pred.size(0)
    if n == 0:
        return {"n": 0}
    mse_per_dim = (se / n).cpu().numpy()
    return {
        "n": n,
        "mse_per_dim": dict(zip(APPRAISAL_DIMS, mse_per_dim.round(4).tolist())),
        "mse_mean": float(mse_per_dim.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="distilbert-base-uncased")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None,
                   help="cut training short for pipeline check")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--output_dir", default="ckpt/appraisal_distilbert")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    print("[data] loading DailyDialog (filtered)...")
    train_ds = EmoFlowDataset("dailydialog", "train",
                              filter_no_emotion_dialogues=True)
    dev_ds = EmoFlowDataset("dailydialog", "dev")
    weights = train_ds.class_weights()
    print(f"[data] train={len(train_ds)}  dev={len(dev_ds)}")
    print(f"[data] class weights (filtered): "
          f"{dict(zip(train_ds.vocab, weights.round(2).tolist()))}")

    def collate_with_weights(batch):
        out = collate_dialogues(batch)
        out["_class_weights"] = weights         # broadcast through to flatten_batch
        return out

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_with_weights)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_with_weights)

    print(f"[model] loading {args.backbone}...")
    model = StimulusEncoder(backbone_name=args.backbone, lora_r=args.lora_r).to(device)
    trainable, total = model.trainable_parameter_count()
    print(f"[model] trainable {trainable:,} / total {total:,}  "
          f"({100*trainable/total:.2f}%)")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            model.train()
            texts, target, weight = flatten_batch(batch, device)
            if not texts:
                continue
            enc = model.encode_text(texts, device)
            pred = model(**enc)
            loss = weighted_mse(pred, target, weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1

            if step % 20 == 0:
                msg = {"step": step, "epoch": epoch, "loss": loss.item(),
                       "n_utt": len(texts), "elapsed": round(time.time()-t0, 1)}
                print(f"  step {step:5d} | loss={loss:.4f} | n={len(texts)}")
                log_f.write(json.dumps(msg) + "\n"); log_f.flush()

            if step % args.eval_every == 0:
                metrics = evaluate(model, dev_loader, device, max_batches=30)
                print(f"  [eval @ step {step}] {metrics}")
                log_f.write(json.dumps({"step": step, "eval": metrics}) + "\n")
                log_f.flush()

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    # final eval (no batch cap)
    print("[final eval] dev set, full pass...")
    metrics = evaluate(model, dev_loader, device)
    print(f"  {metrics}")
    log_f.write(json.dumps({"step": step, "final_eval": metrics}) + "\n")
    log_f.close()

    # save only trainable params (LoRA + head), keyed by their state_dict names
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    ckpt = out_dir / "model.pt"
    torch.save({
        "backbone": args.backbone,
        "lora_r": args.lora_r,
        "trainable_state": {k: v.cpu() for k, v in model.state_dict().items()
                            if k in trainable_names},
        "args": vars(args),
    }, ckpt)
    print(f"[done] saved {ckpt}, log {log_path}")


if __name__ == "__main__":
    main()
