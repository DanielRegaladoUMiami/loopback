"""Popularity-based hard negative sampling.

In-batch negatives are weak: a random batch is dominated by long-tail tracks the
user would never plausibly listen to, so the model only needs to distinguish
"the positive" from "unrelated noise". Hard negatives are tracks that are
*popular* (frequently played in the training set) but not the current positive.
These force the model to learn finer-grained discrimination.

We precompute play counts per `track_idx` from `train.parquet` and use them as
sampling weights with `torch.multinomial(replacement=True)`. To approximately
avoid sampling the positive, we draw N+1 candidates per positive and replace
any collisions with the next-best candidate (rare in practice given 1.5M tracks).
The user's full history is *not* filtered — popularity weighting is a fast
approximation and any collisions act as ~true negatives in expectation.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch


def compute_track_popularity(train_parquet: Path, n_tracks: int) -> torch.Tensor:
    """Return a (n_tracks,) float tensor of play counts (raw, unnormalized).

    `torch.multinomial` accepts unnormalized weights, and we add 1.0 smoothing
    so every track has a non-zero probability of being sampled.
    """
    df = pl.read_parquet(train_parquet)
    counts = (
        df.group_by("track_idx").len()
        .rename({"len": "count"})
        .sort("track_idx")
    )
    weights = torch.ones(n_tracks, dtype=torch.float32)  # +1 smoothing
    idx = torch.from_numpy(counts["track_idx"].to_numpy()).long()
    cnt = torch.from_numpy(counts["count"].to_numpy()).float()
    weights[idx] += cnt
    return weights


class PopularityNegativeSampler:
    """Samples track indices weighted by training-set popularity.

    Holds the weight tensor on CPU (1.5M floats = 6 MB) — `torch.multinomial`
    on MPS for that vocab size is unstable, and CPU sampling is fast enough
    (~few ms per batch of 4096 with N=4 negatives each).
    """

    def __init__(self, weights: torch.Tensor):
        self.weights = weights  # stays on CPU
        self.n_tracks = weights.numel()

    def sample(self, positives: torch.Tensor, n_per_positive: int) -> torch.Tensor:
        """Return a (B, N) long tensor of negative track indices.

        Draws `n_per_positive + 1` candidates per positive then replaces any
        index colliding with the positive using the spare column. This is a
        cheap O(B*N) check that handles the dominant collision case.
        """
        B = positives.numel()
        n_draw = n_per_positive + 1
        # multinomial wants a 1-D weight vector + num_samples; we draw B*n_draw
        # total then reshape. replacement=True keeps it fast.
        flat = torch.multinomial(self.weights, B * n_draw, replacement=True)
        cand = flat.view(B, n_draw)  # (B, N+1)

        pos_cpu = positives.detach().cpu()
        # find collisions: (B, N+1) bool where cand == positive
        collide = cand == pos_cpu.unsqueeze(1)
        # the spare column (last) is the fallback for any collision in cols [0:N]
        negs = cand[:, :n_per_positive].clone()
        spare = cand[:, n_per_positive]
        bad = collide[:, :n_per_positive]
        # broadcast spare across columns of `bad`
        if bad.any():
            # for any (row, col) where bad, replace with spare[row]
            negs = torch.where(bad, spare.unsqueeze(1).expand_as(negs), negs)
        return negs
