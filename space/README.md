---
title: Loopback Two-Tower Music Recommender
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Two-tower music recommender trained on Last.fm 1K
---

# loopback

Open-source two-tower neural recommender trained on Last.fm 1K (15.3M listening events, 1.5M tracks).

- Code: <https://github.com/DanielRegaladoUMiami/loopback>
- Dataset: <https://huggingface.co/datasets/DanielRegaladoCardoso/lastfm-1k-twotower>
- Model: <https://huggingface.co/DanielRegaladoCardoso/loopback-twotower>

## Architecture

```
User tower:  user_id  ──► Embedding ──► MLP ──► L2-norm
Track tower: track_id ──► Embedding ┐
             artist_id ─► Embedding ┴► MLP ──► L2-norm
                                      score = u·t · exp(temp)
```

Trained with symmetric InfoNCE + in-batch negatives (CLIP-style) and a learnable temperature.

## Results

| Metric | Value |
|---|---|
| Recall@10  | 0.0708 |
| Recall@50  | 0.2172 |
| Recall@100 | 0.3140 |

Evaluated on 847 held-out users against the full 1.5M-track catalog with seen-track filtering.
