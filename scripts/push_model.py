"""Push the trained two-tower checkpoint to a HF model repo."""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "DanielRegaladoCardoso/loopback-twotower"
CKPT = Path("checkpoints/two_tower_epoch3.pt")

MODEL_CARD = """---
language:
  - en
license: apache-2.0
library_name: pytorch
tags:
  - recommender-systems
  - two-tower
  - music
  - retrieval
  - contrastive-learning
---

# loopback — two-tower music recommender

Open-source two-tower neural recommender for music, trained from scratch on the
[Last.fm 1K users](https://huggingface.co/datasets/DanielRegaladoCardoso/lastfm-1k-twotower)
dataset. Repo: <https://github.com/DanielRegaladoUMiami/loopback>.

## Architecture

```
User tower:  user_id  ──► Embedding(64) ──► MLP(256→128) ──► L2-norm ──► user_vec
Track tower: track_id ──► Embedding(64) ┐
             artist_id ─► Embedding(64) ┴► MLP(256→128) ──► L2-norm ──► track_vec
                                          score = u · t * exp(temp)
```

Loss: symmetric InfoNCE (CLIP-style) with in-batch negatives and a learnable temperature.

## Training

- 3 epochs, batch size 4096, AdamW lr=1e-3, weight decay 1e-5
- 15.3 M training interactions (992 users × 1.5 M unique tracks)
- Apple M-series MPS, ~9 min / epoch
- Final loss: 5.6 (random baseline at this batch size: ln(4096) ≈ 8.32)

## Results

Evaluated on 847 held-out users with seen-track filtering against the full 1.5 M-track catalog:

| Metric | Value | Random baseline |
|---|---|---|
| Recall@10  | 0.0708 | 6.7 e-6 |
| Recall@50  | 0.2172 | 3.3 e-5 |
| Recall@100 | 0.3140 | 6.7 e-5 |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from loopback.model import TwoTower  # from github.com/DanielRegaladoUMiami/loopback

ckpt = torch.load(hf_hub_download("DanielRegaladoCardoso/loopback-twotower", "two_tower_epoch3.pt"),
                  map_location="cpu", weights_only=False)
model = TwoTower(992, 1_500_661, 174_091, out_dim=ckpt["embed_dim"])
model.load_state_dict(ckpt["model"])
model.eval()
```

## License

Apache 2.0
"""

api = HfApi()
create_repo(REPO_ID, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj=CKPT,
    path_in_repo=CKPT.name,
    repo_id=REPO_ID,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj=MODEL_CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n✅ https://huggingface.co/{REPO_ID}")
