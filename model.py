"""
EmoFlowModel: end-to-end pipeline.

  text         → (frozen LLaMA + LoRA) → appraisal_t (5d)
  appraisal_t  → TemporalMemory(λ)      → memory_state_t (5d)
  (memory_t, appraisal_t) → BayesianHead → posterior_logits (K-dim)

`forward(batch)` takes the collated dict from [[dataloader.py]] and returns
emotion logits (B, T, K), plus the intermediate appraisal vectors (for the
auxiliary appraisal MSE loss).

For pre-Encoder development we ship a MockEncoder that returns random
appraisal vectors per utterance — drop-in compatible with StimulusEncoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from appraisal_targets import APPRAISAL_DIMS
from bayes import BayesianHead
from memory import TemporalMemory

_DEFAULT_APPRAISAL_DIM = len(APPRAISAL_DIMS)


# ---------- MockEncoder for end-to-end testing without LLaMA ----------

class MockEncoder(nn.Module):
    """Random-but-deterministic 5-d appraisal output per utterance.

    Same call surface as StimulusEncoder.encode_text + forward, so the rest
    of the pipeline can be exercised before the real Encoder is trained.
    """

    def __init__(self, appraisal_dim: int = _DEFAULT_APPRAISAL_DIM, seed: int = 0):
        super().__init__()
        self.appraisal_dim = appraisal_dim
        self._gen = torch.Generator().manual_seed(seed)
        # a single learnable scalar so optimizer.parameters() isn't empty
        self.scale = nn.Parameter(torch.ones(1))

    def encode_text(self, texts: list[str], device) -> dict:
        # one "token" per text — Mock doesn't actually tokenize
        return {"texts": texts, "_device": device}

    def forward(self, *, texts: list[str], _device, **_):
        n = len(texts)
        a = torch.rand(n, self.appraisal_dim, generator=self._gen).to(_device)
        return a * self.scale       # keeps gradient flowing for optimizer.step()


# ---------- EmoFlowModel ----------

class EmoFlowModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_emotions: int,
        appraisal_dim: int = _DEFAULT_APPRAISAL_DIM,
        init_lambda: float = 0.1,
        freeze_lambda: bool = False,
        bayes_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.memory = TemporalMemory(init_lambda=init_lambda,
                                     freeze_lambda=freeze_lambda)
        self.bayes = BayesianHead(appraisal_dim=appraisal_dim,
                                  num_emotions=num_emotions,
                                  hidden=bayes_hidden)

    def _encode_dialogues(self, batch: dict, device) -> torch.Tensor:
        """Flatten (B, T) → (N_valid,), encode, scatter back to (B, T, 5)."""
        utt_mask = batch["utt_mask"]                            # (B, T) bool np
        flat_idx = np.argwhere(utt_mask)                        # (N, 2)
        texts = [batch["text"][b][t] for b, t in flat_idx]

        # chunk to avoid OOM on long dialogues with big encoders
        CHUNK = 8
        chunks = []
        for i in range(0, len(texts), CHUNK):
            enc = self.encoder.encode_text(texts[i:i+CHUNK], device)
            chunks.append(self.encoder(**enc))
        flat_appraisal = torch.cat(chunks, dim=0)               # (N, D)

        B, T = utt_mask.shape
        D = flat_appraisal.size(-1)
        appraisal = flat_appraisal.new_zeros(B, T, D)
        for k, (b, t) in enumerate(flat_idx):
            appraisal[b, t] = flat_appraisal[k]
        return appraisal

    def forward(self, batch: dict, device=None) -> dict:
        if device is None:
            device = next(self.parameters()).device

        appraisal = self._encode_dialogues(batch, device)        # (B, T, 5)
        turn_idx = torch.from_numpy(batch["turn_idx"]).to(device)
        utt_mask = torch.from_numpy(batch["utt_mask"]).to(device)

        memory_state = self.memory(appraisal, turn_idx, utt_mask)
        bayes = self.bayes(memory_state, appraisal)

        return {
            "appraisal":        appraisal,                       # (B, T, 5)
            "memory_state":     memory_state,                    # (B, T, 5)
            "prior_logits":     bayes["prior_logits"],           # (B, T, K)
            "likelihood_logits":bayes["likelihood_logits"],
            "posterior_logits": bayes["posterior_logits"],
            "utt_mask":         utt_mask,                        # (B, T) bool
        }


if __name__ == "__main__":
    import torch.nn.functional as F

    from dataloader import EmoFlowDataset, collate_dialogues
    print("=== EmoFlowModel end-to-end smoke test (MockEncoder) ===")
    torch.manual_seed(0)
    ds = EmoFlowDataset("meld", "train")
    batch = collate_dialogues([ds[i] for i in range(3)])
    print(f"  batch shapes: utt_mask={batch['utt_mask'].shape}, "
          f"target={batch['appraisal_target'].shape}")

    model = EmoFlowModel(encoder=MockEncoder(), num_emotions=len(ds.vocab))
    out = model(batch)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:20s}: shape={tuple(v.shape)} dtype={v.dtype}")

    # joint loss: emotion CE + appraisal MSE (with mask)
    label = torch.from_numpy(batch["emotion_label_idx"]).to(out["posterior_logits"].device)
    utt_mask = out["utt_mask"]
    K = out["posterior_logits"].size(-1)
    ce = F.cross_entropy(
        out["posterior_logits"].reshape(-1, K), label.reshape(-1),
        reduction="none",
    ) * utt_mask.reshape(-1).float()
    ce = ce.sum() / utt_mask.sum().clamp(min=1)

    appraisal_target = torch.from_numpy(batch["appraisal_target"]).to(out["appraisal"].device)
    appraisal_mask = torch.from_numpy(batch["appraisal_mask"]).to(out["appraisal"].device)
    mse_per = ((out["appraisal"] - appraisal_target) ** 2).mean(-1)   # (B, T)
    mse = (mse_per * appraisal_mask).sum() / appraisal_mask.sum().clamp(min=1)

    total = ce + 0.5 * mse
    total.backward()
    print(f"\n  CE loss  = {ce.item():.4f}")
    print(f"  MSE loss = {mse.item():.4f}")
    print(f"  total backward OK (grads flow through encoder.scale: "
          f"{model.encoder.scale.grad is not None})")
    print(f"  λ value  = {model.memory.lam.item():.4f}  "
          f"(grad: {model.memory._lambda_raw.grad is not None})")
