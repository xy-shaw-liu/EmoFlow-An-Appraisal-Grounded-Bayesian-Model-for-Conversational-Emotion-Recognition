"""
StimulusEncoder + AppraisalHead.

Architecture (per EmoFlow project structure D3-D5):
  text  →  frozen backbone (Meta-Llama-3-8B; locally distilbert for dev)
       →  LoRA-adapted attention
       →  pooled utterance representation h ∈ R^d
       →  AppraisalHead: Linear(d → d) → GELU → Linear(d → 8)   (NO sigmoid; see §6)
       →  appraisal vector ∈ R^8  (regressed toward Scherer Table 5.5 [0,1] targets via MSE)

Backbone is frozen; only LoRA adapters + AppraisalHead are trained. Swap
`backbone_name="meta-llama/Meta-Llama-3-8B"` after HF gated access lands.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from appraisal_targets import APPRAISAL_DIMS

_DEFAULT_APPRAISAL_DIM = len(APPRAISAL_DIMS)


# Backbones we support; values list LoRA target_modules for the attention.
# distilbert: 6 layers, 768 dim — fast local dev.
# llama-3-8b: 32 layers, 4096 dim — final training target.
_BACKBONES = {
    "distilbert-base-uncased": {
        "target_modules": ["q_lin", "v_lin"],
        "pool": "cls",
    },
    "roberta-base": {
        "target_modules": ["query", "value"],
        "pool": "cls",
    },
    "meta-llama/Meta-Llama-3-8B": {
        "target_modules": ["q_proj", "v_proj"],
        "pool": "last",
    },
}


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    """Reduce (B, T, d) to (B, d)."""
    if how == "cls":
        return hidden[:, 0]                                    # [CLS] token
    if how == "last":
        # last non-pad token per sequence
        lengths = attention_mask.sum(dim=1) - 1                # (B,)
        return hidden[torch.arange(hidden.size(0)), lengths]
    if how == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(how)


class StimulusEncoder(nn.Module):
    """Frozen backbone + LoRA + AppraisalHead. Produces 8-d appraisal vector."""

    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        appraisal_dim: int = _DEFAULT_APPRAISAL_DIM,
    ):
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone_name}; "
                             f"add it to _BACKBONES (need target_modules, pool)")
        cfg = _BACKBONES[backbone_name]
        self.pool_how = cfg["pool"]

        # 4-bit QLoRA for LLaMA-class models (frees ~10 GB on A10 24GB);
        # plain bf16 + LoRA for smaller backbones used in local dev.
        if backbone_name.startswith(("meta-llama/", "mistralai/")):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            backbone = AutoModel.from_pretrained(
                backbone_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
            )
            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=True,
            )
        else:
            backbone = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.bfloat16)
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=cfg["target_modules"],
            bias="none", task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        hidden = backbone.config.hidden_size
        # NOTE: no sigmoid on output. Earlier sigmoid-bounded version suffered
        # catastrophic saturation (encoder output became bit-identical across
        # all inputs, std=0.000). Unbounded output preserves gradient flow;
        # MSE against [0,1] Scherer targets still pulls outputs into range.
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, appraisal_dim),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ----- forward path -----

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """input_ids/attention_mask: (B, T). Returns (B, 8) appraisal vector."""
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False)
        h = _pool(out.last_hidden_state, attention_mask, self.pool_how)
        return self.head(h.float())

    def encode_text(self, texts: list[str], device, max_length: int = 96) -> dict:
        """Tokenize a list of utterance strings on `device`."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        # only the two fields forward() consumes — drop token_type_ids etc.
        return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}

    # ----- diagnostics -----

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


if __name__ == "__main__":
    print("=== StimulusEncoder smoke test (distilbert) ===")
    torch.manual_seed(0)
    model = StimulusEncoder(backbone_name="distilbert-base-uncased")
    trainable, total = model.trainable_parameter_count()
    print(f"  trainable params: {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.2f}%)")

    texts = [
        "I'm so happy to see you!",
        "This is really frustrating.",
        "What a surprise.",
    ]
    enc = model.encode_text(texts, device="cpu")
    print(f"  tokenized: input_ids={tuple(enc['input_ids'].shape)}")

    with torch.no_grad():
        out = model(**enc)
    print(f"  output appraisal: {tuple(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")
    print(f"  sample (happy):     {out[0].tolist()}")
    print(f"  sample (frustrate): {out[1].tolist()}")
    print(f"  sample (surprise):  {out[2].tolist()}")
