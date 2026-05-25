"""
Unified training loop for EmoFlow + baselines on EmoryNLP or MELD.

  python3 train.py --model emoflow   --dataset emorynlp
  python3 train.py --model stateless --dataset emorynlp
  python3 train.py --model lstm      --dataset meld
  python3 train.py --model emoflow   --dataset emorynlp \
                   --freeze_lambda --init_lambda 0          # λ ablation
  python3 train.py --model emoflow   --dataset emorynlp \
                   --init_ckpt ckpt/appraisal_distilbert/model.pt  # warm-start

Joint objective:   L = CE(emotion) + α · MSE(appraisal, mask)
                   both terms use inverse-frequency class weights computed
                   on the training split.

Output:
  ckpt/<run>/args.json
            /train_log.jsonl
            /best_model.pt        — params at best dev weighted-F1
            /metrics.json         — final test set (weighted-F1, ETA, etc.)
            /predictions.csv      — dialogue_id, turn_idx, y_true, y_pred
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from baselines import LSTMMemoryModel, StatelessClassifier
from dataloader import EmoFlowDataset, collate_dialogues
from encoder import StimulusEncoder
from evaluation import compute_metrics
from model import EmoFlowModel, MockEncoder


def build_model(args, num_emotions: int, device) -> nn.Module:
    if args.backbone == "mock":
        encoder = MockEncoder()
    else:
        encoder = StimulusEncoder(backbone_name=args.backbone, lora_r=args.lora_r)
        if args.init_ckpt:
            ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = encoder.load_state_dict(
                ckpt["trainable_state"], strict=False)
            print(f"[init] loaded {args.init_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

    if args.model == "emoflow":
        model = EmoFlowModel(encoder=encoder, num_emotions=num_emotions,
                             init_lambda=args.init_lambda,
                             freeze_lambda=args.freeze_lambda)
    elif args.model == "stateless":
        model = StatelessClassifier(encoder=encoder, num_emotions=num_emotions)
    elif args.model == "lstm":
        model = LSTMMemoryModel(encoder=encoder, num_emotions=num_emotions)
    else:
        raise ValueError(args.model)
    return model.to(device)


def _build_label_remap(vocab: list[str]):
    """For --multilabel mode: split vocab into (vocab_6, neutral_idx7, remap7to6).

    remap7to6[i] = -1 if vocab[i]=='neutral' else index in vocab_6.
    """
    if "neutral" not in vocab:
        raise ValueError(f"--multilabel needs 'neutral' in vocab; got {vocab}")
    neutral_idx7 = vocab.index("neutral")
    vocab_6 = [v for v in vocab if v != "neutral"]
    remap7to6 = [-1 if v == "neutral" else vocab_6.index(v) for v in vocab]
    return vocab_6, neutral_idx7, remap7to6


def joint_loss(out: dict, batch: dict, device,
               class_weights: torch.Tensor,
               appraisal_alpha: float,
               args_label_smoothing: float = 0.0,
               multilabel: bool = False,
               remap7to6_t: torch.Tensor | None = None,
               bce_pos_weight: torch.Tensor | None = None) -> dict:
    logits = out["posterior_logits"]                         # (B, T, K)
    K = logits.size(-1)
    label = torch.from_numpy(batch["emotion_label_idx"]).to(device)  # (B, T) in 7-class space
    utt_mask = out["utt_mask"]                                # (B, T) bool

    if multilabel:
        # 6-dim multi-hot target; neutral turn → all zeros.
        label6 = remap7to6_t[label]                              # (B, T), -1 for neutral
        target_ml = torch.zeros(*label.shape, K, device=device, dtype=logits.dtype)
        valid = label6 >= 0
        if valid.any():
            target_ml[valid] = F.one_hot(label6[valid], num_classes=K).to(logits.dtype)
        bce_per = F.binary_cross_entropy_with_logits(
            logits, target_ml, reduction="none",
            pos_weight=bce_pos_weight,
        ).mean(-1)                                            # (B, T)
        bce_per = bce_per * utt_mask.float()
        ce = bce_per.sum() / utt_mask.sum().clamp(min=1)
        sample_w = torch.ones_like(label, dtype=torch.float32)
    else:
        ce_per = F.cross_entropy(
            logits.reshape(-1, K), label.reshape(-1),
            weight=class_weights, reduction="none",
            label_smoothing=args_label_smoothing,
        )
        ce_per = ce_per * utt_mask.reshape(-1).float()
        ce = ce_per.sum() / utt_mask.sum().clamp(min=1)
        sample_w = class_weights[label]                       # (B, T)

    # appraisal MSE (only where mask=True)
    appraisal_target = torch.from_numpy(batch["appraisal_target"]).to(device)
    appraisal_mask = torch.from_numpy(batch["appraisal_mask"]).to(device)
    mse_per = ((out["appraisal"] - appraisal_target) ** 2).mean(-1)  # (B, T)
    mse_per = mse_per * sample_w
    denom = (sample_w * appraisal_mask).sum().clamp(min=1)
    mse = (mse_per * appraisal_mask).sum() / denom

    return {"loss": ce + appraisal_alpha * mse, "ce": ce, "mse": mse}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> dict:
    model.eval()
    rows_did, rows_t, rows_yt, rows_yp = [], [], [], []
    for batch in loader:
        out = model(batch, device=device)
        pred = out["posterior_logits"].argmax(dim=-1).cpu().numpy()   # (B, T)
        utt_mask = batch["utt_mask"]
        label = batch["emotion_label_idx"]
        for b in range(utt_mask.shape[0]):
            for t in range(utt_mask.shape[1]):
                if utt_mask[b, t]:
                    rows_did.append(batch["dialogue_id"][b])
                    rows_t.append(int(batch["turn_idx"][b, t]))
                    rows_yt.append(int(label[b, t]))
                    rows_yp.append(int(pred[b, t]))
    return {"dialogue_id": rows_did, "turn_idx": rows_t,
            "y_true": rows_yt, "y_pred": rows_yp}


@torch.no_grad()
def predict_multilabel(model: nn.Module, loader: DataLoader, device,
                       threshold: float, neutral_idx7: int,
                       idx6_to_idx7: np.ndarray) -> dict:
    """Predictions returned in original 7-class index space.

    Per turn: sigmoid(6 logits); if max prob < threshold → predict neutral,
    else predict argmax (remapped to 7-class space).
    """
    model.eval()
    rows_did, rows_t, rows_yt, rows_yp = [], [], [], []
    for batch in loader:
        out = model(batch, device=device)
        probs = torch.sigmoid(out["posterior_logits"]).cpu().numpy()  # (B, T, 6)
        max_p = probs.max(-1)                                          # (B, T)
        argmax6 = probs.argmax(-1)                                     # (B, T)
        pred7 = idx6_to_idx7[argmax6]
        pred7 = np.where(max_p < threshold, neutral_idx7, pred7)
        utt_mask = batch["utt_mask"]
        label = batch["emotion_label_idx"]
        for b in range(utt_mask.shape[0]):
            for t in range(utt_mask.shape[1]):
                if utt_mask[b, t]:
                    rows_did.append(batch["dialogue_id"][b])
                    rows_t.append(int(batch["turn_idx"][b, t]))
                    rows_yt.append(int(label[b, t]))
                    rows_yp.append(int(pred7[b, t]))
    return {"dialogue_id": rows_did, "turn_idx": rows_t,
            "y_true": rows_yt, "y_pred": rows_yp}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["emoflow", "stateless", "lstm"])
    p.add_argument("--dataset", required=True, choices=["emorynlp", "meld"])
    p.add_argument("--backbone", default="distilbert-base-uncased",
                   help="HF model id; or 'mock' for MockEncoder")
    p.add_argument("--init_ckpt", default=None,
                   help="optional pretrained appraisal encoder checkpoint")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--appraisal_alpha", type=float, default=0.5)
    p.add_argument("--no_class_weights", action="store_true",
                   help="disable inverse-freq class weights in CE + MSE")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="CE label smoothing (0.1 typical)")
    p.add_argument("--init_lambda", type=float, default=0.1)
    p.add_argument("--freeze_lambda", action="store_true",
                   help="for λ ablation (set with --init_lambda 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_train_dialogues", type=int, default=None,
                   help="cap training set size for fast debug")
    p.add_argument("--oversample", action="store_true",
                   help="WeightedRandomSampler over dialogues (inverse-freq) "
                        "to break majority-class collapse")
    p.add_argument("--dd_rare", default=None,
                   help="comma-sep rare classes to inject from DailyDialog "
                        "train. e.g. 'fear,disgust,sadness'. Adds DD dialogues "
                        "containing ANY listed class to train_ds.")
    p.add_argument("--multilabel", action="store_true",
                   help="Train as 6-way multi-label BCE; neutral = all-zero "
                        "target. Inference: sigmoid + threshold → neutral.")
    p.add_argument("--threshold_sweep", default="0.2,0.3,0.4,0.5,0.6",
                   help="comma-sep dev thresholds to try (multilabel only)")
    p.add_argument("--bce_pos_weight", action="store_true",
                   help="Apply per-dim pos_weight = (1-p)/p to BCE so that "
                        "rare classes can't be ignored by marginal-output. "
                        "Requires --multilabel.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- data ----
    train_ds = EmoFlowDataset(args.dataset, "train")
    dev_ds = EmoFlowDataset(args.dataset, "dev")
    test_ds = EmoFlowDataset(args.dataset, "test")
    if args.max_train_dialogues:
        train_ds._dialogues = train_ds._dialogues[:args.max_train_dialogues]
    if args.dd_rare and args.dataset != "dailydialog":
        rare_set = set(s.strip() for s in args.dd_rare.split(","))
        unknown = rare_set - set(train_ds.vocab)
        if unknown:
            raise ValueError(f"--dd_rare classes {unknown} not in vocab {train_ds.vocab}")
        dd = EmoFlowDataset("dailydialog", "train", filter_no_emotion_dialogues=True)
        if dd.vocab != train_ds.vocab:
            raise ValueError(f"DD vocab {dd.vocab} ≠ train vocab {train_ds.vocab}")
        dd_added = [
            d for d in dd._dialogues
            if {dd.vocab[u.emotion_label_idx] for u in d.utterances} & rare_set
        ]
        print(f"[dd_rare] +{len(dd_added)} DailyDialog dialogues "
              f"(filter classes: {sorted(rare_set)})")
        train_ds._dialogues = train_ds._dialogues + dd_added
    K_data = len(train_ds.vocab)
    if args.multilabel:
        vocab_6, neutral_idx7, remap7to6 = _build_label_remap(train_ds.vocab)
        K_model = len(vocab_6)
        remap7to6_t = torch.tensor(remap7to6, dtype=torch.long, device=device)
        idx6_to_idx7 = np.array([train_ds.vocab.index(v) for v in vocab_6],
                                dtype=np.int64)
        weights = None
        if args.bce_pos_weight:
            # per-dim positive rate over ALL train turns (neutral = all zeros)
            counts7 = np.zeros(K_data, dtype=np.float64)
            n_turns = 0
            for d in train_ds._dialogues:
                for u in d.utterances:
                    counts7[u.emotion_label_idx] += 1
                    n_turns += 1
            p_pos = np.array(
                [counts7[train_ds.vocab.index(v)] / n_turns for v in vocab_6],
                dtype=np.float32,
            )
            pw = (1.0 - p_pos) / np.clip(p_pos, 1e-6, None)
            bce_pos_weight_t = torch.from_numpy(pw).to(device)
            print(f"[bce_pos_weight] p_pos={dict(zip(vocab_6, p_pos.round(3).tolist()))}")
            print(f"[bce_pos_weight]      pw={dict(zip(vocab_6, pw.round(2).tolist()))}")
        else:
            bce_pos_weight_t = None
        print(f"[multilabel] K_model=6  vocab_6={vocab_6}  "
              f"neutral_idx7={neutral_idx7}")
    else:
        K_model = K_data
        remap7to6_t = None
        idx6_to_idx7 = None
        neutral_idx7 = None
        bce_pos_weight_t = None
        if args.no_class_weights:
            weights = torch.ones(K_model, device=device)
            print("[loss] class_weights disabled (uniform=1)")
        else:
            weights = torch.from_numpy(train_ds.class_weights()).to(device)
    print(f"[data] {args.dataset}: train={len(train_ds)} dev={len(dev_ds)} "
          f"test={len(test_ds)} K_data={K_data} K_model={K_model} "
          f"vocab={train_ds.vocab}")

    if args.oversample:
        cw_np = train_ds.class_weights()
        dialogue_weights = np.array([
            sum(float(cw_np[u.emotion_label_idx]) for u in d.utterances)
            for d in train_ds._dialogues
        ], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=dialogue_weights.tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        print(f"[sampler] WeightedRandomSampler over {len(dialogue_weights)} "
              f"dialogues, weight range "
              f"[{dialogue_weights.min():.2f}, {dialogue_weights.max():.2f}]")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, collate_fn=collate_dialogues)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_dialogues)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dialogues)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dialogues)

    # ---- model + optimizer ----
    model = build_model(args, K_model, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model} on {args.backbone}  trainable={trainable:,}")
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # ---- output ----
    run = args.run_name or (f"{args.model}_{args.dataset}_"
                            f"{args.backbone.replace('/', '-')}_s{args.seed}")
    out_dir = Path("ckpt") / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = (out_dir / "train_log.jsonl").open("w")

    # ---- train loop ----
    best_dev_f1 = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            model.train()
            out = model(batch, device=device)
            losses = joint_loss(out, batch, device, weights, args.appraisal_alpha,
                                args.label_smoothing,
                                multilabel=args.multilabel, remap7to6_t=remap7to6_t,
                                bce_pos_weight=bce_pos_weight_t)
            optim.zero_grad()
            losses["loss"].backward()
            optim.step()
            step += 1
            if step % 20 == 0:
                msg = {"step": step, "epoch": epoch,
                       "loss": losses["loss"].item(),
                       "ce": losses["ce"].item(),
                       "mse": losses["mse"].item(),
                       "elapsed": round(time.time() - t0, 1)}
                print(f"  step {step:5d} | loss={msg['loss']:.3f} "
                      f"ce={msg['ce']:.3f} mse={msg['mse']:.3f}")
                log_f.write(json.dumps(msg) + "\n"); log_f.flush()
            if args.max_steps and step >= args.max_steps:
                break

        # end of epoch eval — multilabel uses default threshold=0.5 here;
        # final threshold is swept on dev after training (below).
        if args.multilabel:
            pred = predict_multilabel(model, dev_loader, device,
                                      threshold=0.5,
                                      neutral_idx7=neutral_idx7,
                                      idx6_to_idx7=idx6_to_idx7)
        else:
            pred = predict(model, dev_loader, device)
        m = compute_metrics(pred["dialogue_id"], pred["turn_idx"],
                            pred["y_true"], pred["y_pred"])
        m["epoch"] = epoch
        print(f"  [dev @ ep {epoch}] wF1={m['weighted_f1']:.4f}  "
              f"ETA={m['eta']:.4f} ({m['n_correct']}/{m['n_transitions']})")
        log_f.write(json.dumps({"dev_eval": m}) + "\n"); log_f.flush()
        if m["weighted_f1"] > best_dev_f1:
            best_dev_f1 = m["weighted_f1"]
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            slim_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
            torch.save({"state_dict": slim_state,
                        "args": vars(args), "dev_metric": m},
                       out_dir / "best_model.pt")
            print(f"  ★ new best dev wF1={best_dev_f1:.4f}, saved")
        if args.max_steps and step >= args.max_steps:
            break

    # ---- final test eval ----
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    if args.multilabel:
        # Sweep threshold on dev with best ckpt, then apply to test.
        thresholds = [float(t) for t in args.threshold_sweep.split(",")]
        best_t, best_f1 = thresholds[0], -1.0
        for t in thresholds:
            p_dev = predict_multilabel(model, dev_loader, device, threshold=t,
                                       neutral_idx7=neutral_idx7,
                                       idx6_to_idx7=idx6_to_idx7)
            f1_t = compute_metrics(p_dev["dialogue_id"], p_dev["turn_idx"],
                                   p_dev["y_true"], p_dev["y_pred"])["weighted_f1"]
            print(f"  [threshold sweep] t={t:.2f}  dev wF1={f1_t:.4f}")
            if f1_t > best_f1:
                best_t, best_f1 = t, f1_t
        print(f"  ★ best threshold = {best_t:.2f}  (dev wF1={best_f1:.4f})")
        pred = predict_multilabel(model, test_loader, device, threshold=best_t,
                                  neutral_idx7=neutral_idx7,
                                  idx6_to_idx7=idx6_to_idx7)
    else:
        best_t = None
        pred = predict(model, test_loader, device)
    m = compute_metrics(pred["dialogue_id"], pred["turn_idx"],
                        pred["y_true"], pred["y_pred"],
                        label_names=train_ds.vocab, verbose=True)
    # Also report 6-way wF1 (exclude neutral turns) for direct comparison
    # with DialogueRNN/COSMIC "without-neutral" protocol.
    if "neutral" in train_ds.vocab:
        n_idx = train_ds.vocab.index("neutral")
        y_t6 = [yt for yt in pred["y_true"] if yt != n_idx]
        y_p6 = [yp for yt, yp in zip(pred["y_true"], pred["y_pred"]) if yt != n_idx]
        from evaluation import weighted_f1 as _wf1
        m["weighted_f1_no_neutral"] = _wf1(y_t6, y_p6)
        m["n_eval_no_neutral"] = len(y_t6)
    if best_t is not None:
        m["best_threshold"] = best_t
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"[test] wF1={m['weighted_f1']:.4f}  ETA={m['eta']:.4f}"
          + (f"  wF1_6way={m['weighted_f1_no_neutral']:.4f}"
             if 'weighted_f1_no_neutral' in m else ''))

    # predictions.csv
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dialogue_id", "turn_idx", "y_true_idx", "y_pred_idx",
                    "y_true", "y_pred"])
        for did, t, yt, yp in zip(pred["dialogue_id"], pred["turn_idx"],
                                  pred["y_true"], pred["y_pred"]):
            w.writerow([did, t, yt, yp, train_ds.vocab[yt], train_ds.vocab[yp]])

    log_f.close()
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
