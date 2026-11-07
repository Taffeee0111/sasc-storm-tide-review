from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class STGCNLite(nn.Module):
    def __init__(self, dynamic_in_dim: int, static_in_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal_conv1 = nn.Conv1d(dynamic_in_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.adj_blend_logit = nn.Parameter(torch.tensor(0.0))
        self.self_loop_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, dynamic_x: torch.Tensor, static_s: torch.Tensor, adj_phys: torch.Tensor) -> torch.Tensor:
        # dynamic_x: [B, N, T, F], static_s: [N, S], adj_phys: [N, N]
        bsz, num_nodes, lookback, feat_dim = dynamic_x.shape
        temporal = dynamic_x.reshape(bsz * num_nodes, lookback, feat_dim).permute(0, 2, 1)
        temporal = F.relu(self.temporal_conv1(temporal))
        temporal = F.relu(self.temporal_conv2(temporal))
        temporal = temporal[:, :, -1]
        temporal = temporal.view(bsz, num_nodes, -1)
        temporal = self.temporal_norm(temporal)

        static_emb = self.static_encoder(static_s).unsqueeze(0).expand(bsz, num_nodes, -1)
        node_state = F.relu(self.mix(torch.cat([temporal, static_emb], dim=-1)))

        eye = torch.eye(num_nodes, device=adj_phys.device, dtype=adj_phys.dtype)
        blend = torch.sigmoid(self.adj_blend_logit)
        self_loop = torch.sigmoid(self.self_loop_logit)
        adj = blend * adj_phys + (1.0 - blend) * eye
        adj = adj + self_loop * eye
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        propagated = torch.einsum('ij,bjh->bih', adj, node_state)
        propagated = F.relu(self.gcn(propagated))
        propagated = self.dropout(propagated)
        out = self.decoder(torch.cat([propagated, static_emb], dim=-1)).squeeze(-1)
        return out
