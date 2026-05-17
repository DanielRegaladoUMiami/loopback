"""Train the two-tower model on Last.fm 1K interactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loopback.model import TwoTower, info_nce_loss

PROCESSED_DIR = Path("data/processed")
CKPT_DIR = Path("checkpoints")


def load_split(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pl.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return (
        torch.from_numpy(df["user_idx"].to_numpy()).long(),
        torch.from_numpy(df["track_idx"].to_numpy()).long(),
        torch.from_numpy(df["artist_idx"].to_numpy()).long(),
    )


def train(
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    embed_dim: int = 128,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    vocab = pl.read_parquet(PROCESSED_DIR / "vocab.parquet").row(0, named=True)
    n_users, n_tracks, n_artists = vocab["n_users"], vocab["n_tracks"], vocab["n_artists"]

    train_data = TensorDataset(*load_split("train"))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = TwoTower(n_users, n_tracks, n_artists, out_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        for u, t, a in pbar:
            u, t, a = u.to(device), t.to(device), a.to(device)
            logits = model(u, t, a)
            loss = info_nce_loss(logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * u.size(0)
            count += u.size(0)
            pbar.set_postfix(loss=total / count, temp=model.log_temp.exp().item())
        torch.save(
            {"model": model.state_dict(), "vocab": vocab, "embed_dim": embed_dim},
            CKPT_DIR / f"two_tower_epoch{epoch+1}.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.embed_dim)


if __name__ == "__main__":
    main()
