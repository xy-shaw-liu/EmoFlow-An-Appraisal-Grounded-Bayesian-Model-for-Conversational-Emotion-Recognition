# EmoFlow

**An Appraisal-Grounded Bayesian Model for Conversational Emotion Recognition**

ECS 271 Final Project — UC Davis, Spring 2026.

EmoFlow factorizes conversational emotion recognition (ERC) into three stages mirroring Scherer's Component Process Model: a frozen LLaMA-3-8B encoder with LoRA adapters maps each utterance to an 8-dimensional **appraisal vector**, a **TemporalMemory** module aggregates past appraisals via a learnable exponential-decay kernel, and a **BayesianHead** fuses prior (memory) and likelihood (current utterance) into per-turn emotion logits.

**Headline result:** wF1 = 0.62 on MELD test, matching DialogueRNN with ~20M trainable parameters.

---

## Architecture

```
Text ──[StimulusEncoder]──▶ Appraisal (8d) ──[TemporalMemory]──▶ Memory state
                                  │                                  │
                                  ▼                                  ▼
                          [BayesianHead: log-additive fusion of likelihood + prior]
                                                │
                                                ▼
                                          Emotion logits
```

Three design choices set EmoFlow apart from standard ERC models:

1. **Appraisal grounding** — the bottleneck is interpretable: each of the 8 dimensions corresponds to a Scherer appraisal check (novelty, intrinsic pleasantness, goal conduciveness, control, …), weakly supervised against Scherer 2001 Table 5.5.
2. **Bayesian fusion** — context-driven prior and utterance-driven likelihood are kept separable instead of being entangled inside an RNN hidden state.
3. **Exponential-decay memory** — temporal weighting via a single learnable λ rather than a generic recurrent cell.

See [REPORT.md](REPORT.md) for the full method and results, and [emoflow_algorithm_explained.md](emoflow_algorithm_explained.md) for a zero-prerequisite walkthrough with all derivations.

---

## Repository layout

| File | Purpose |
|---|---|
| `preprocess.py` | Unifies EmoryNLP / MELD / DailyDialog into a common JSONL `Dialogue` schema |
| `dataloader.py` | `EmoFlowDataset` + dialogue-level collate |
| `encoder.py` | `StimulusEncoder` — backbone (DistilBERT / LLaMA) + LoRA + AppraisalHead |
| `memory.py` | `TemporalMemory` with learnable exponential decay |
| `bayes.py` | `BayesianHead` — log-additive prior/likelihood fusion |
| `model.py` | `EmoFlowModel` wiring everything together |
| `baselines.py` | `StatelessClassifier`, `LSTMMemoryModel` |
| `appraisal_targets.py` | Scherer 2001 Table 5.5 → per-emotion 8-d appraisal targets |
| `train.py` | Unified training loop (EmoFlow + baselines, EmoryNLP / MELD) |
| `train_appraisal.py` | Stage-1 appraisal-only pretraining on DailyDialog |
| `evaluation.py` | Weighted-F1 / macro-F1 / per-class metrics |
| `make_table.py` | Builds the results table from `ckpt/*/metrics.json` |
| `setup_lambda.sh`, `overnight.sh` | Lambda Labs provisioning and unattended sweep |
| `REPORT.md` | Final report |
| `emoflow_algorithm_explained.md` | Algorithm walkthrough (Chinese) |

---

## Setup

```bash
pip install torch transformers peft datasets scikit-learn
# Optional: bitsandbytes for QLoRA on the LLaMA backbone
pip install bitsandbytes
```

Tested with Python 3.10, PyTorch 2.x, CUDA 12 on a single A100 / H100.

---

## Data

The repo does **not** ship raw datasets. Place or download them under `datasets/`:

- **MELD** — primary benchmark. Clone from https://github.com/declare-lab/MELD into `datasets/MELD/`.
- **EmoryNLP** — secondary benchmark.
- **DailyDialog** — used for stage-1 appraisal pretraining (HuggingFace `daily_dialog`).

Then run:

```bash
python preprocess.py            # writes data_processed/{meld,emorynlp,dailydialog}.jsonl
```

---

## Training

**Main MELD run** — the exact config behind the wF1 = 0.62 headline (see `ckpt_lambda/emoflow_meld_ddrare3_os/args.json`):

```bash
python train.py --model emoflow --dataset meld \
                --backbone meta-llama/Meta-Llama-3-8B \
                --multilabel --dd_rare fear,disgust,sadness --oversample \
                --epochs 3 --batch_size 2 --lr 5e-4 \
                --appraisal_alpha 0.1 --init_lambda 0.1 --seed 42
```

For a fast smoke test without GPU access, swap in `--backbone distilbert-base-uncased`.

**Baselines:**

```bash
python train.py --model stateless --dataset meld
python train.py --model lstm      --dataset meld
```

**λ ablation** (freeze the memory decay rate):

```bash
python train.py --model emoflow --dataset meld --freeze_lambda --init_lambda 0
```

**Warm-start from appraisal pretraining:**

```bash
python train_appraisal.py --dataset dailydialog            # stage 1
python train.py --model emoflow --dataset meld \
                --init_ckpt ckpt/appraisal_distilbert/model.pt
```

Outputs land in `ckpt/<run_name>/`:

```
args.json            run configuration
train_log.jsonl      per-step train/dev metrics
best_model.pt        params at best dev weighted-F1
metrics.json         final test wF1, macro-F1, ETA
predictions.csv      dialogue_id, turn_idx, y_true, y_pred
```

---

## Reproducing the results table

The four rows of the main table (REPORT §7.1) come from these runs — all share
`--dataset meld --multilabel --dd_rare fear,disgust,sadness --oversample --epochs 3 --batch_size 2 --lr 5e-4 --appraisal_alpha 0.1`:

```bash
# EmoFlow (learned λ)  → wF1 0.6171
python train.py --model emoflow  --backbone meta-llama/Meta-Llama-3-8B [shared flags] --init_lambda 0.1
# EmoFlow (λ=0, no decay)  → wF1 0.6052
python train.py --model emoflow  --backbone meta-llama/Meta-Llama-3-8B [shared flags] --init_lambda 0 --freeze_lambda
# Stateless baseline  → wF1 0.5631
python train.py --model stateless --backbone meta-llama/Meta-Llama-3-8B [shared flags]
# LSTM baseline  → wF1 0.4241
python train.py --model lstm      --backbone meta-llama/Meta-Llama-3-8B [shared flags]

python make_table.py             # collates ckpt_lambda/*/metrics.json into the results table
```

The MELD wF1 = 0.62 figure corresponds to the EmoFlow (learned λ) run with the **sigmoid-saturation fix** (see REPORT §6) and rare-class augmentation from DailyDialog enabled via `--dd_rare`.

> Note: `overnight.sh` is an earlier EmoryNLP-targeted sweep and does **not** reproduce the MELD headline; use the commands above.

---

## What's excluded from this repo

To keep the repo lean, the following are gitignored:

- `ckpt/`, `ckpt_lambda/` — model checkpoints (the 1.6 GB Lambda bundle is kept locally)
- `data_processed/` — regenerated by `preprocess.py`
- `datasets/MELD/`, `datasets/emotion-detection/` — pull from their upstream repos
- `logs_lambda/`, `__pycache__/`

---

## References

- Scherer, K. R. (2001). *Appraisal Considered as a Process of Multilevel Sequential Checking.* In *Appraisal Processes in Emotion.*
- Scherer, K. R. (1997). *Profiles of emotion-antecedent appraisal: Testing theoretical predictions across cultures.* ISEAR, 37 countries.
- Poria et al. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.*
- Majumder et al. (2019). *DialogueRNN.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*

Full bibliography in [REPORT.md](REPORT.md).