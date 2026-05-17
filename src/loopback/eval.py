"""Evaluate two-tower with recall@K using FAISS retrieval."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from loopback.model import TwoTower

PROCESSED_DIR = Path("data/processed")
CKPT_DIR = Path("checkpoints")


def build_index(model: TwoTower, n_tracks: int, artist_lookup: np.ndarray, device: str) -> faiss.Index:
    """Embed every track and put it in FAISS."""
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for start in tqdm(range(0, n_tracks, 8192), desc="indexing tracks"):
            end = min(start + 8192, n_tracks)
            t = torch.arange(start, end, device=device)
            a = torch.from_numpy(artist_lookup[start:end]).long().to(device)
            v = model.track_tower(t, a).cpu().numpy()
            all_vecs.append(v)
    vecs = np.concatenate(all_vecs).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on L2-normalized = cosine
    index.add(vecs)
    return index


def evaluate(ckpt_path: Path, ks: tuple[int, ...] = (10, 50, 100)) -> dict[int, float]:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]

    model = TwoTower(vocab["n_users"], vocab["n_tracks"], vocab["n_artists"], out_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)

    train_df = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    artist_lookup = (
        train_df.group_by("track_idx").agg(pl.col("artist_idx").first())
        .sort("track_idx")["artist_idx"].to_numpy()
    )
    if len(artist_lookup) != vocab["n_tracks"]:
        full = np.zeros(vocab["n_tracks"], dtype=np.int64)
        for row in train_df.group_by("track_idx").agg(pl.col("artist_idx").first()).iter_rows():
            full[row[0]] = row[1]
        artist_lookup = full

    index = build_index(model, vocab["n_tracks"], artist_lookup, device)

    user_history: dict[int, set[int]] = defaultdict(set)
    for u, t in train_df.select("user_idx", "track_idx").iter_rows():
        user_history[u].add(t)

    test_df = pl.read_parquet(PROCESSED_DIR / "test.parquet")
    test_positives: dict[int, set[int]] = defaultdict(set)
    for u, t in test_df.select("user_idx", "track_idx").iter_rows():
        test_positives[u].add(t)

    eval_users = [u for u in test_positives if u in user_history]
    max_k = max(ks)

    hits = {k: 0 for k in ks}
    counts = {k: 0 for k in ks}
    with torch.no_grad():
        for u in tqdm(eval_users, desc="recall"):
            uv = model.user_tower(torch.tensor([u], device=device)).cpu().numpy().astype("float32")
            _, topk = index.search(uv, max_k + len(user_history[u]))
            seen = user_history[u]
            filtered = [t for t in topk[0] if t not in seen][:max_k]
            pos = test_positives[u]
            for k in ks:
                hits[k] += len(set(filtered[:k]) & pos) > 0
                counts[k] += 1

    return {k: hits[k] / counts[k] for k in ks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    args = parser.parse_args()
    ckpt = args.ckpt or sorted(CKPT_DIR.glob("two_tower_epoch*.pt"))[-1]
    print(f"evaluating {ckpt}")
    results = evaluate(ckpt)
    for k, r in results.items():
        print(f"recall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
