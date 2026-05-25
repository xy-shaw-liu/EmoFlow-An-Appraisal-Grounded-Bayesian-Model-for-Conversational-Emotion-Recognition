#!/bin/bash
# Overnight runner for the full EmoFlow result chain.
# Assumes setup_lambda.sh has been run successfully on this instance.
#
#   tmux new -s overnight
#   ./overnight.sh
#   # Ctrl+B then D to detach. Reattach with: tmux attach -t overnight

set -e
cd "$(dirname "$0")"
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] overnight start"
echo "[$(ts)] python: $(which python3) | nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- 0. quick smoke test on real LLaMA — 3 steps, no harm if it fails ----
echo "[$(ts)] === SMOKE TEST (real LLaMA, 3 steps, ~2-3 min) ==="
python3 train.py --model emoflow --dataset emorynlp \
    --backbone meta-llama/Meta-Llama-3-8B \
    --max_train_dialogues 4 --epochs 1 --max_steps 3 \
    --batch_size 2 --run_name _llama_smoke \
    2>&1 | tee logs/00_smoke.log
rm -rf ckpt/_llama_smoke
echo "[$(ts)] smoke test passed."

# ---- 1. DailyDialog appraisal pretraining ----
echo "[$(ts)] === APPRAISAL PRETRAINING on DailyDialog ==="
python3 train_appraisal.py \
    --backbone meta-llama/Meta-Llama-3-8B \
    --batch_size 4 --epochs 1 --max_steps 3000 \
    --output_dir ckpt/appraisal_llama3 \
    2>&1 | tee logs/01_appraisal.log

# ---- 2. EmoFlow full ----
echo "[$(ts)] === EMOFLOW FULL (with λ learned) ==="
python3 train.py --model emoflow --dataset emorynlp \
    --backbone meta-llama/Meta-Llama-3-8B \
    --init_ckpt ckpt/appraisal_llama3/model.pt \
    --batch_size 4 --epochs 3 --seed 42 \
    2>&1 | tee logs/02_emoflow.log

# ---- 3. EmoFlow λ ablation ----
echo "[$(ts)] === EMOFLOW λ=0 ABLATION (no decay) ==="
python3 train.py --model emoflow --dataset emorynlp \
    --backbone meta-llama/Meta-Llama-3-8B \
    --init_ckpt ckpt/appraisal_llama3/model.pt \
    --init_lambda 0 --freeze_lambda \
    --batch_size 4 --epochs 3 --seed 42 \
    2>&1 | tee logs/03_emoflow_nodecay.log

# ---- 4. Stateless baseline ----
echo "[$(ts)] === STATELESS BASELINE ==="
python3 train.py --model stateless --dataset emorynlp \
    --backbone meta-llama/Meta-Llama-3-8B \
    --batch_size 4 --epochs 3 --seed 42 \
    2>&1 | tee logs/04_stateless.log

# ---- 5. LSTM baseline ----
echo "[$(ts)] === LSTM BASELINE ==="
python3 train.py --model lstm --dataset emorynlp \
    --backbone meta-llama/Meta-Llama-3-8B \
    --batch_size 4 --epochs 3 --seed 42 \
    2>&1 | tee logs/05_lstm.log

# ---- 6. Aggregate ----
echo "[$(ts)] === AGGREGATE ==="
python3 make_table.py 2>&1 | tee logs/06_table.log
cat results.md

echo "[$(ts)] === ALL DONE ==="
