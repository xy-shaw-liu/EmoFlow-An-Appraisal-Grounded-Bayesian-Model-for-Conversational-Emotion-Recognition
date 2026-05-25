"""
TemporalMemory: exp(−λ·Δt) weighted aggregator over past appraisal vectors.

For a sequence of per-turn appraisal vectors a_1..a_T with turn indices τ_1..τ_T,
the memory state at turn t is

    h_t = Σ_{i≤t} w_i · a_i / Σ_{i≤t} w_i
    w_i = exp(−λ · (τ_t − τ_i))

λ ≥ 0 is a learnable scalar (parameterized via softplus for positivity).
λ = 0 → uniform average over all past + current (the "no decay" ablation
branch the project plans to compare against).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMemory(nn.Module):
    def __init__(self, init_lambda: float = 0.1, freeze_lambda: bool = False):
        super().__init__()
        # store the pre-softplus param so softplus(_lambda_raw) = λ
        # softplus^{-1}(x) = log(exp(x) - 1)
        x = max(init_lambda, 1e-4)
        inv = torch.log(torch.expm1(torch.tensor(x)))
        self._lambda_raw = nn.Parameter(inv, requires_grad=not freeze_lambda)

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self._lambda_raw)

    def forward(
        self,
        appraisal: torch.Tensor,    # (B, T, D)  per-turn appraisal vectors
        turn_idx: torch.Tensor,     # (B, T)     0-based turn indices (long)
        utt_mask: torch.Tensor,     # (B, T)     bool, True for real utterances
    ) -> torch.Tensor:
        B, T, D = appraisal.shape
        lam = self.lam

        # Δt[b, t, i] = turn_idx[b, t] − turn_idx[b, i]
        t_row = turn_idx.unsqueeze(2).float()                  # (B, T, 1)
        t_col = turn_idx.unsqueeze(1).float()                  # (B, 1, T)
        dt = t_row - t_col                                     # (B, T, T)

        # only attend to past + current real utterances
        causal = (dt >= 0)                                     # (B, T, T)
        valid = utt_mask.unsqueeze(1) & causal                 # (B, T, T)

        # log-weights (subtract max for numerical stability), then softmax-like
        logw = -lam * dt
        logw = logw.masked_fill(~valid, float("-inf"))
        w = torch.softmax(logw, dim=-1)                        # (B, T, T)

        # rows where t_row corresponds to a padding slot stay all-NaN (no valid
        # keys) — convert to zeros to keep downstream computations finite
        w = torch.nan_to_num(w, nan=0.0)

        h = torch.bmm(w, appraisal)                            # (B, T, D)
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 5
    appraisal = torch.randn(B, T, D)
    turn_idx = torch.arange(T).unsqueeze(0).repeat(B, 1)
    utt_mask = torch.tensor([[1, 1, 1, 1, 0],   # last is padding
                             [1, 1, 1, 1, 1]], dtype=torch.bool)

    print("=== TemporalMemory smoke test ===")
    for init in [0.0, 0.1, 5.0]:
        m = TemporalMemory(init_lambda=init)
        h = m(appraisal, turn_idx, utt_mask)
        print(f"  λ_init={init}  λ={m.lam.item():.4f}  h[0,2,:2]={h[0,2,:2].tolist()}")

    # check ablation: λ→0 should give running mean over past
    m0 = TemporalMemory(init_lambda=1e-4)
    h0 = m0(appraisal[:1], turn_idx[:1], utt_mask[:1])
    expected = appraisal[0, :3].mean(0)
    print(f"  λ≈0  h[0,2]    = {h0[0,2].tolist()}")
    print(f"       mean(a0..2)= {expected.tolist()}")
