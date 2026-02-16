from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDisc(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(256, d_model)
        self.pad_id = pad_id  # <--- store pad id

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model + 15, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, features16: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.tok(input_ids) + self.pos(pos)

        # Build padding mask: True where we want to ignore positions
        pad_mask = (input_ids == self.pad_id)  # [B, T], True for PAD

        # Let the transformer ignore PADs in attention
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Masked mean pool: only average over non-PAD tokens
        nonpad = (~pad_mask).float()              # 1 for real tokens, 0 for PAD
        x_sum = (x * nonpad.unsqueeze(-1)).sum(1) # [B, d_model]
        lengths = nonpad.sum(1, keepdim=True)     # [B, 1]
        pooled = x_sum / (lengths + 1e-9)         # [B, d_model]

        z = torch.cat([pooled, features16], dim=-1)
        logit = self.fc(z).squeeze(-1)
        return logit
