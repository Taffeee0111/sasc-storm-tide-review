"""A compact, audited BiLSTM-Attention regression backbone."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMAttention(nn.Module):
    """Encode a causal input window and predict the following water level."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = int(input_size)
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, features]")
        if inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} input features, "
                f"received {inputs.shape[-1]}"
            )
        encoded, _ = self.encoder(inputs)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.head(context).squeeze(-1)
