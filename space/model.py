"""Two-tower neural recommender.

User tower:  user_id ──► Embedding ──► MLP ──► L2-normalized user_vec
Track tower: (track_id, artist_id) ──► Embeddings ──► MLP ──► L2-normalized track_vec

Score = dot(user_vec, track_vec). Trained with InfoNCE / in-batch negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class UserTower(nn.Module):
    def __init__(self, n_users: int, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.proj = mlp(embed_dim, 256, out_dim)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        return F.normalize(self.proj(x), dim=-1)


class TrackTower(nn.Module):
    def __init__(self, n_tracks: int, n_artists: int, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.proj = mlp(embed_dim * 2, 256, out_dim)

    def forward(self, track_idx: torch.Tensor, artist_idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.track_emb(track_idx), self.artist_emb(artist_idx)], dim=-1)
        return F.normalize(self.proj(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_tracks: int, n_artists: int, out_dim: int = 128):
        super().__init__()
        self.user_tower = UserTower(n_users, out_dim=out_dim)
        self.track_tower = TrackTower(n_tracks, n_artists, out_dim=out_dim)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # learnable temperature, like CLIP

    def forward(
        self, user_idx: torch.Tensor, track_idx: torch.Tensor, artist_idx: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_tower(user_idx)  # (B, D)
        t = self.track_tower(track_idx, artist_idx)  # (B, D)
        logits = (u @ t.T) * self.log_temp.exp()  # (B, B)
        return logits


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE — diagonal is the positive pair, off-diagonal are in-batch negatives."""
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
