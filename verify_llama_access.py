"""
Verify HuggingFace gated access to Meta-Llama-3-8B works.

Run this once before scheduling the overnight job. If gated access hasn't
been granted yet, this will fail fast (HTTPError 403) rather than letting
the smoke test in overnight.sh discover it at 1am.

  python3 verify_llama_access.py
"""

import sys

from huggingface_hub import HfApi


MODEL = "meta-llama/Meta-Llama-3-8B"


def main():
    api = HfApi()
    try:
        info = api.model_info(MODEL)
    except Exception as e:
        print(f"FAIL: cannot read model info for {MODEL}")
        print(f"      {type(e).__name__}: {e}")
        print("\nLikely causes:")
        print("  - HF gated access not yet granted (check email / model page)")
        print("  - `huggingface-cli login` not run on this machine")
        print("  - HF_TOKEN env var not set")
        sys.exit(1)

    print(f"OK : model_info accessible for {MODEL}")
    print(f"     last modified: {info.lastModified}")
    print(f"     # files: {len(info.siblings)}")

    # Try the actual tokenizer download (cheap, ~5MB)
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        print(f"OK : tokenizer loaded (vocab size {tok.vocab_size:,})")
    except Exception as e:
        print(f"FAIL: tokenizer load failed: {e}")
        sys.exit(1)

    print("\nGated access works. You can run overnight.sh tonight.")
    print("Tip: to pre-download the 16GB weights (so smoke test is faster):")
    print("  python3 -c \"from transformers import AutoModel; "
          f"AutoModel.from_pretrained('{MODEL}')\"")


if __name__ == "__main__":
    main()
