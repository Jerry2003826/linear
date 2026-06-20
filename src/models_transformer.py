from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.encoder(h, mask=self.causal_mask[:seq_len, :seq_len])
        h = self.norm(h)
        return self.lm_head(h)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
