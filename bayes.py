"""
BayesianHead: log-additive posterior over emotion classes.

At each turn t,
    prior_logits      = prior_head(memory_state_t)      # context-driven
    likelihood_logits = lik_head(current_appraisal_t)   # utterance-driven
    posterior_logits  = prior_logits + likelihood_logits

In log-space this is Bayes' rule under a uniform marginal:
    log p(e|x, ctx) = log p(x|e) + log p(e|ctx) − log Z
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BayesianHead(nn.Module):
    def __init__(self, appraisal_dim: int, num_emotions: int,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()

        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(appraisal_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_emotions),
            )
        self.prior_head = mlp()
        self.likelihood_head = mlp()

    def forward(
        self,
        memory_state: torch.Tensor,    # (B, T, D)
        current_appraisal: torch.Tensor,  # (B, T, D)
    ) -> dict[str, torch.Tensor]:
        prior_logits = self.prior_head(memory_state)
        lik_logits = self.likelihood_head(current_appraisal)
        posterior_logits = prior_logits + lik_logits
        return {
            "prior_logits": prior_logits,
            "likelihood_logits": lik_logits,
            "posterior_logits": posterior_logits,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    head = BayesianHead(appraisal_dim=5, num_emotions=7)
    B, T, D = 2, 4, 5
    memory = torch.randn(B, T, D)
    current = torch.randn(B, T, D)
    out = head(memory, current)
    print("=== BayesianHead smoke test ===")
    for k, v in out.items():
        print(f"  {k:18s}: shape={tuple(v.shape)} "
              f"range=[{v.min():.2f}, {v.max():.2f}]")
    probs = torch.softmax(out["posterior_logits"], dim=-1)
    print(f"  posterior probs sum (should be 1.0): {probs.sum(dim=-1)[0,0].item():.4f}")
