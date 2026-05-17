# loopback

Two-tower neural recommender for music, built from scratch in PyTorch.

## What this is

An open-source implementation of the canonical **two-tower retrieval architecture** used in production at YouTube, Spotify, and other recommenders. Trained on the Last.fm 1K users dataset (19M listening events).

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

## License

Apache 2.0
