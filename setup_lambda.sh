#!/bin/bash
# One-time setup on a fresh Lambda instance with persistent filesystem attached.
# Run from /home/ubuntu/persistent/EmoFlow/ after the code has been rsync'd in.
#
#   bash setup_lambda.sh

set -e
cd "$(dirname "$0")"
echo "[setup] working dir: $(pwd)"

echo "[setup] installing python deps..."
pip install -q -U \
    torch \
    transformers \
    peft \
    accelerate \
    scikit-learn \
    datasets \
    pypdf \
    pymupdf \
    huggingface_hub

echo "[setup] python deps installed."

if [ -z "$HF_TOKEN" ] && ! [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "[setup] no HuggingFace login found."
    echo "        run: huggingface-cli login"
    echo "        and paste your token, then re-run this script."
    exit 1
fi

echo "[setup] verifying LLaMA-3-8B gated access..."
python3 verify_llama_access.py

echo "[setup] mini smoke test (mock backbone, ~30s)..."
python3 train.py --model emoflow --dataset emorynlp --backbone mock \
    --max_train_dialogues 4 --epochs 1 --max_steps 2 \
    --run_name _setup_smoke

rm -rf ckpt/_setup_smoke
echo
echo "[setup] all green. ready for overnight.sh"
