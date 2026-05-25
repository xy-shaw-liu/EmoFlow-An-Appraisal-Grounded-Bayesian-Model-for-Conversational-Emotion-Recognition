"""
Two baselines for comparing against EmoFlow.

  StatelessClassifier
      Encoder → Linear → emotion logits.
      No memory, no Bayes. Each turn predicted independently.
      Isolates the contribution of temporal modeling.

  LSTMMemoryModel
      Encoder → BiLSTM over per-turn appraisals → Linear → emotion logits.
      Standard temporal baseline (DialogueRNN-style without the speaker GRU).
      Apples-to-apples replacement for TemporalMemory + BayesianHead.

Both take an `encoder` (StimulusEncoder, MockEncoder, or anything with the
same `encode_text` + forward surface) and integrate with [[dataloader.py]] /
[[evaluation.py]] just like [[model.py]].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from appraisal_targets import APPRAISAL_DIMS

_DEFAULT_APPRAISAL_DIM = len(APPRAISAL_DIMS)


# Helper shared with EmoFlowModel — flatten utterances, encode, scatter back.
def _encode_dialogues(encoder: nn.Module, batch: dict, device,
                      appraisal_dim: int) -> torch.Tensor:
    utt_mask = batch["utt_mask"]                              # (B, T) bool np
    flat_idx = np.argwhere(utt_mask)                          # (N, 2)
    texts = [batch["text"][b][t] for b, t in flat_idx]
    CHUNK = 8
    chunks = []
    for i in range(0, len(texts), CHUNK):
        enc = encoder.encode_text(texts[i:i+CHUNK], device)
        chunks.append(encoder(**enc))
    flat = torch.cat(chunks, dim=0)                            # (N, D)
    B, T = utt_mask.shape
    appraisal = flat.new_zeros(B, T, appraisal_dim)
    for k, (b, t) in enumerate(flat_idx):
        appraisal[b, t] = flat[k]
    return appraisal


# ---------------------------------------------------------------------------

class StatelessClassifier(nn.Module):
    """Per-turn classification, no temporal context."""

    def __init__(self, encoder: nn.Module, num_emotions: int,
                 appraisal_dim: int = _DEFAULT_APPRAISAL_DIM, hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        self.appraisal_dim = appraisal_dim
        self.head = nn.Sequential(
            nn.Linear(appraisal_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_emotions),
        )

    def forward(self, batch: dict, device=None) -> dict:
        if device is None:
            device = next(self.parameters()).device
        appraisal = _encode_dialogues(self.encoder, batch, device, self.appraisal_dim)
        logits = self.head(appraisal)                         # (B, T, K)
        utt_mask = torch.from_numpy(batch["utt_mask"]).to(device)
        return {"posterior_logits": logits, "appraisal": appraisal,
                "utt_mask": utt_mask}


# ---------------------------------------------------------------------------

class LSTMMemoryModel(nn.Module):
    """BiLSTM over per-turn appraisals as the temporal memory."""

    def __init__(self, encoder: nn.Module, num_emotions: int,
                 appraisal_dim: int = _DEFAULT_APPRAISAL_DIM, lstm_hidden: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.appraisal_dim = appraisal_dim
        self.lstm = nn.LSTM(
            input_size=appraisal_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_emotions),
        )

    def forward(self, batch: dict, device=None) -> dict:
        if device is None:
            device = next(self.parameters()).device
        appraisal = _encode_dialogues(self.encoder, batch, device, self.appraisal_dim)
        # naive: feed padded sequence (padding rows are zeros — bias is tiny
        # since utt_mask gates the loss anyway). If perf matters, switch to
        # pack_padded_sequence using utt_mask.sum(dim=1) as lengths.
        h, _ = self.lstm(appraisal)                            # (B, T, 2H)
        logits = self.head(h)                                  # (B, T, K)
        utt_mask = torch.from_numpy(batch["utt_mask"]).to(device)
        return {"posterior_logits": logits, "appraisal": appraisal,
                "utt_mask": utt_mask}


if __name__ == "__main__":
    from dataloader import EmoFlowDataset, collate_dialogues
    from model import MockEncoder

    print("=== Baseline smoke test (MockEncoder, 3 MELD dialogues) ===")
    ds = EmoFlowDataset("meld", "train")
    batch = collate_dialogues([ds[i] for i in range(3)])

    for name, cls in [("StatelessClassifier", StatelessClassifier),
                      ("LSTMMemoryModel",     LSTMMemoryModel)]:
        model = cls(encoder=MockEncoder(), num_emotions=len(ds.vocab))
        out = model(batch)
        logits = out["posterior_logits"]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            torch.from_numpy(batch["emotion_label_idx"]).reshape(-1),
        )
        loss.backward()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:22s}: logits={tuple(logits.shape)} "
              f"loss={loss.item():.3f} trainable={n_trainable:,}")
