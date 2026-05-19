# loopback

Two-tower neural recommender for music, built from scratch in PyTorch.

## What this is

An open-source implementation of the canonical **two-tower retrieval architecture** used in production at YouTube, Spotify, and other recommenders. Trained on the Last.fm 1K users dataset (19M listening events).

> **Paper-style writeup:** [`docs/PAPER.md`](docs/PAPER.md) — abstract, method, math, results, limitations.

The goal: a reference implementation that's *readable*, *reproducible*, and *honest about the math* — contrastive loss, in-batch negatives, sampled softmax, and FAISS retrieval explained from first principles.

## Architecture

```
User tower:    user_id  ──► Embedding ──► MLP ──► user_vec  (dim=128)
                                                       │
                                          dot product  │
                                                       │
Track tower:   track_id ──► Embedding ──► MLP ──► track_vec (dim=128)
                + artist_id, audio features
```

Trained with **InfoNCE contrastive loss** using in-batch negatives. At inference, all track vectors are indexed in **FAISS** for sub-millisecond top-K retrieval.

## Quickstart

```bash
uv sync
uv run python -m loopback.data prepare    # download + process Last.fm 1K
uv run python -m loopback.train           # train two-tower
uv run python -m loopback.eval            # recall@10, recall@50
```

## Dataset

[Last.fm 1K users](http://mtg.upf.edu/node/1671) — 19M listening events from 1000 users (2005-2009). Schema: `(user_id, timestamp, artist_mbid, artist_name, track_mbid, track_name)`.

Temporal split: oldest 80% train, next 10% val, last 10% test.

After processing: **992 users · 1.5M unique tracks · 174K artists · 19.1M interactions** (15.3M train / 1.9M val / 1.9M test).

## Results

3 epochs, batch 4096, embed dim 128, in-batch negatives, AdamW lr=1e-3, Apple M-series MPS, ~9 min/epoch.

| Metric | Value | Random baseline |
|---|---|---|
| Recall@10  | **0.0708** | 6.7e-6 |
| Recall@50  | **0.2172** | 3.3e-5 |
| Recall@100 | **0.3140** | 6.7e-5 |

Evaluated on 847 test users with seen-track filtering against the full 1.5M-track catalog.

## Math: why InfoNCE with in-batch negatives works

Given a batch of `B` (user, positive-track) pairs, we compute the `B × B` similarity matrix `S` where `S[i,j] = u_i · t_j`. The diagonal is the positive pair, off-diagonal entries are treated as negatives — every other track in the batch is an "implicit" negative for user `i`.

The loss is symmetric cross-entropy on `S` (user→track) and `S.T` (track→user), exactly the CLIP objective. With a learnable temperature, the model learns how peaked the softmax should be. This avoids ever materializing the full 1.5M-track softmax denominator that would make naive maximum-likelihood training infeasible.

## License

Apache 2.0
