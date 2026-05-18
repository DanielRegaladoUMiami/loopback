"""Push processed Last.fm 1K parquet to HF Hub as a dataset."""

import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "DanielRegaladoCardoso/lastfm-1k-twotower"
PROCESSED = Path("data/processed")

api = HfApi()
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
print("repo ready", flush=True)

for f in ["README.md", "train.parquet", "val.parquet", "test.parquet", "vocab.parquet", "track_labels.parquet"]:
    p = PROCESSED / f
    size = p.stat().st_size / 1e6
    print(f"uploading {f} ({size:.1f} MB)...", flush=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=p,
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({size/dt:.1f} MB/s)", flush=True)

print(f"\n✅ https://huggingface.co/datasets/{REPO_ID}", flush=True)
