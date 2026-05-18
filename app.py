"""Loopback — Gradio demo for the two-tower music recommender.

Pick 3 tracks → the app creates a synthetic user vector by averaging their embeddings,
then retrieves the top-10 nearest tracks with FAISS.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import gradio as gr
import numpy as np
import polars as pl
import torch
from huggingface_hub import hf_hub_download

from loopback.model import TwoTower

MODEL_REPO = "DanielRegaladoCardoso/loopback-twotower"
DATASET_REPO = "DanielRegaladoCardoso/lastfm-1k-twotower"
CKPT_NAME = "two_tower_epoch3.pt"


def load_artifacts():
    ckpt_path = hf_hub_download(MODEL_REPO, CKPT_NAME)
    train_path = hf_hub_download(DATASET_REPO, "train.parquet", repo_type="dataset")
    vocab_path = hf_hub_download(DATASET_REPO, "vocab.parquet", repo_type="dataset")

    vocab = pl.read_parquet(vocab_path).row(0, named=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = TwoTower(vocab["n_users"], vocab["n_tracks"], vocab["n_artists"], out_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_df = pl.read_parquet(train_path)

    # (track_idx → artist_idx) lookup
    artist_lookup = np.zeros(vocab["n_tracks"], dtype=np.int64)
    for row in train_df.group_by("track_idx").agg(pl.col("artist_idx").first()).iter_rows():
        artist_lookup[row[0]] = row[1]

    # Embed every track and index in FAISS
    all_vecs = []
    with torch.no_grad():
        for start in range(0, vocab["n_tracks"], 8192):
            end = min(start + 8192, vocab["n_tracks"])
            t = torch.arange(start, end)
            a = torch.from_numpy(artist_lookup[start:end]).long()
            all_vecs.append(model.track_tower(t, a).numpy())
    vecs = np.concatenate(all_vecs).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    labels_path = hf_hub_download(DATASET_REPO, "track_labels.parquet", repo_type="dataset")
    labels_df = pl.read_parquet(labels_path).sort("track_idx")
    labels = labels_df["label"].to_list()

    return model, index, labels, vecs


print("Loading model + index from HF Hub (cold start can take ~30s)...")
MODEL, INDEX, LABELS, TRACK_VECS = load_artifacts()


LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def recommend(track_choices: list[str], k: int = 10) -> str:
    if not track_choices:
        return "Pick at least one track."
    idxs = [LABEL_TO_IDX[c] for c in track_choices if c in LABEL_TO_IDX]
    if not idxs:
        return "None of those tracks matched. Try another."
    seed_vec = TRACK_VECS[idxs].mean(axis=0, keepdims=True)
    seed_vec = seed_vec / np.linalg.norm(seed_vec)
    scores, neighbors = INDEX.search(seed_vec.astype("float32"), k + len(idxs))
    out = [(n, s) for n, s in zip(neighbors[0], scores[0]) if n not in idxs][:k]
    return "\n".join(f"{i+1}. {LABELS[n]}  (sim={s:.3f})" for i, (n, s) in enumerate(out))


# Sample of popular-ish tracks for the dropdown — full 1.5M would freeze the UI.
# Use uniform stride for variety; Gradio also supports type-to-search.
SAMPLE_TRACKS = sorted(LABELS[::500][:2000])

with gr.Blocks(title="loopback") as demo:
    gr.Markdown("# loopback — two-tower music recommender\nPick 3 tracks you'd listen to. The model averages their embeddings and retrieves the 10 nearest neighbors from a 1.5M-track catalog (Last.fm 1K).")
    picks = gr.Dropdown(choices=SAMPLE_TRACKS, multiselect=True, label="Seed tracks")
    btn = gr.Button("Recommend")
    out = gr.Textbox(label="Top-10 recommendations", lines=12)
    btn.click(recommend, inputs=picks, outputs=out)

if __name__ == "__main__":
    demo.launch()
