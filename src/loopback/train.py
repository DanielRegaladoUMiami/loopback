"""Train the two-tower model on Last.fm 1K interactions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loopback.model import TwoTower, info_nce_loss
from loopback.sampling import PopularityNegativeSampler, compute_track_popularity

PROCESSED_DIR = Path("data/processed")
CKPT_DIR = Path("checkpoints")


def load_split(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pl.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return (
        torch.from_numpy(df["user_idx"].to_numpy()).long(),
        torch.from_numpy(df["track_idx"].to_numpy()).long(),
        torch.from_numpy(df["artist_idx"].to_numpy()).long(),
    )


def build_artist_lookup(n_tracks: int) -> torch.Tensor:
    """track_idx -> artist_idx, taking the first artist seen per track in train."""
    df = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    g = df.group_by("track_idx").agg(pl.col("artist_idx").first()).sort("track_idx")
    lookup = np.zeros(n_tracks, dtype=np.int64)
    idx = g["track_idx"].to_numpy()
    val = g["artist_idx"].to_numpy()
    lookup[idx] = val
    return torch.from_numpy(lookup).long()


def hard_negative_logits(
    model: TwoTower,
    u_vec: torch.Tensor,           # (B, D) already normalized
    t_vec: torch.Tensor,           # (B, D) positives, normalized
    neg_track_idx: torch.Tensor,   # (B*N,) on device
    neg_artist_idx: torch.Tensor,  # (B*N,) on device
    temp: torch.Tensor,
) -> torch.Tensor:
    """Return (B, B + B*N) logits: in-batch block then hard-neg block."""
    n_vec = model.track_tower(neg_track_idx, neg_artist_idx)  # (B*N, D)
    in_batch = u_vec @ t_vec.T          # (B, B)
    extra = u_vec @ n_vec.T             # (B, B*N)
    return torch.cat([in_batch, extra], dim=1) * temp


def info_nce_with_hard_negs(logits_full: torch.Tensor, logits_inbatch: torch.Tensor) -> torch.Tensor:
    """User->track uses (B, B+B*N); track->user uses only the in-batch (B,B) block.

    The hard negatives are sampled per-user-positive — they aren't valid as
    "candidate users" for the track->user direction, so we keep that direction
    symmetric on the in-batch slice only. This matches how sampled-softmax is
    usually applied: extra negatives only enter the user->track softmax.
    """
    B = logits_full.size(0)
    targets = torch.arange(B, device=logits_full.device)
    return 0.5 * (
        F.cross_entropy(logits_full, targets)
        + F.cross_entropy(logits_inbatch.T, targets)
    )


def train(
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    embed_dim: int = 128,
    hard_negatives: int = 0,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} hard_negatives={hard_negatives}")

    vocab = pl.read_parquet(PROCESSED_DIR / "vocab.parquet").row(0, named=True)
    n_users, n_tracks, n_artists = vocab["n_users"], vocab["n_tracks"], vocab["n_artists"]

    train_data = TensorDataset(*load_split("train"))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = TwoTower(n_users, n_tracks, n_artists, out_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = None
    artist_lookup = None
    if hard_negatives > 0:
        print("precomputing track popularity...")
        weights = compute_track_popularity(PROCESSED_DIR / "train.parquet", n_tracks)
        sampler = PopularityNegativeSampler(weights)
        print("building artist lookup...")
        artist_lookup = build_artist_lookup(n_tracks)  # stays on CPU; we index then .to(device)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_prefix = "two_tower_hardneg" if hard_negatives > 0 else "two_tower"

    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        epoch_start = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        for u, t, a in pbar:
            u_dev = u.to(device, non_blocking=True)
            t_dev = t.to(device, non_blocking=True)
            a_dev = a.to(device, non_blocking=True)

            if hard_negatives > 0:
                # sample on CPU using the positives, then move to device
                negs = sampler.sample(t, hard_negatives)         # (B, N) cpu
                negs_flat = negs.reshape(-1)                     # (B*N,)
                neg_artists = artist_lookup[negs_flat]           # (B*N,) cpu
                negs_flat = negs_flat.to(device, non_blocking=True)
                neg_artists = neg_artists.to(device, non_blocking=True)

                u_vec = model.user_tower(u_dev)
                t_vec = model.track_tower(t_dev, a_dev)
                temp = model.log_temp.exp()
                logits_inbatch = (u_vec @ t_vec.T) * temp
                logits_full = hard_negative_logits(
                    model, u_vec, t_vec, negs_flat, neg_artists, temp
                )
                loss = info_nce_with_hard_negs(logits_full, logits_inbatch)
            else:
                logits = model(u_dev, t_dev, a_dev)
                loss = info_nce_loss(logits)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * u.size(0)
            count += u.size(0)
            pbar.set_postfix(loss=total / count, temp=model.log_temp.exp().item())

        dur = time.time() - epoch_start
        print(f"epoch {epoch+1} done in {dur:.1f}s, avg loss={total/count:.4f}")
        torch.save(
            {"model": model.state_dict(), "vocab": vocab, "embed_dim": embed_dim,
             "hard_negatives": hard_negatives},
            CKPT_DIR / f"{ckpt_prefix}_epoch{epoch+1}.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hard-negatives", type=int, default=0,
                        help="N hard negatives per positive (0 = in-batch only)")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.embed_dim, hard_negatives=args.hard_negatives)


if __name__ == "__main__":
    main()
