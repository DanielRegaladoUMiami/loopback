"""Download and process the Last.fm 1K users dataset into train-ready parquet."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

DATA_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TSV_NAME = "userid-timestamp-artid-artname-traid-traname.tsv"

TSV_SCHEMA = {
    "user_id": pl.Utf8,
    "timestamp": pl.Utf8,
    "artist_mbid": pl.Utf8,
    "artist_name": pl.Utf8,
    "track_mbid": pl.Utf8,
    "track_name": pl.Utf8,
}


def download(url: str = DATA_URL, dest_dir: Path = RAW_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "lastfm-dataset-1K.tar.gz"
    if archive.exists():
        print(f"[skip] {archive} already exists")
        return archive

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(archive, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    return archive


def extract(archive: Path, dest_dir: Path = RAW_DIR) -> Path:
    tsv = dest_dir / TSV_NAME
    if tsv.exists():
        print(f"[skip] {tsv} already extracted")
        return tsv
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(TSV_NAME):
                member.name = TSV_NAME
                tar.extract(member, dest_dir)
                break
    return tsv


def process(tsv: Path, out_dir: Path = PROCESSED_DIR) -> None:
    """Clean, encode IDs, temporal split, write parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pl.scan_csv(
            tsv,
            separator="\t",
            has_header=False,
            new_columns=list(TSV_SCHEMA.keys()),
            schema_overrides=TSV_SCHEMA,
            ignore_errors=True,
        )
        .drop_nulls(["user_id", "track_name", "artist_name", "timestamp"])
        .with_columns(
            pl.col("timestamp").str.to_datetime(strict=False).alias("ts"),
            pl.concat_str(["artist_name", "track_name"], separator=" — ").alias("track_key"),
        )
        .drop_nulls("ts")
        .collect(streaming=True)
    )

    user_ids = df["user_id"].unique().sort()
    track_keys = df["track_key"].unique().sort()
    artist_names = df["artist_name"].unique().sort()
    user_map = {u: i for i, u in enumerate(user_ids)}
    track_map = {t: i for i, t in enumerate(track_keys)}
    artist_map = {a: i for i, a in enumerate(artist_names)}

    df = df.with_columns(
        pl.col("user_id").replace_strict(user_map).cast(pl.Int32).alias("user_idx"),
        pl.col("track_key").replace_strict(track_map).cast(pl.Int32).alias("track_idx"),
        pl.col("artist_name").replace_strict(artist_map).cast(pl.Int32).alias("artist_idx"),
    ).sort("ts")

    n = df.height
    train = df.slice(0, int(n * 0.8))
    val = df.slice(int(n * 0.8), int(n * 0.1))
    test = df.slice(int(n * 0.9), n - int(n * 0.9))

    cols = ["user_idx", "track_idx", "artist_idx", "ts"]
    train.select(cols).write_parquet(out_dir / "train.parquet")
    val.select(cols).write_parquet(out_dir / "val.parquet")
    test.select(cols).write_parquet(out_dir / "test.parquet")

    vocab = pl.DataFrame(
        {
            "n_users": [len(user_map)],
            "n_tracks": [len(track_map)],
            "n_artists": [len(artist_map)],
        }
    )
    vocab.write_parquet(out_dir / "vocab.parquet")

    print(f"users={len(user_map):,} tracks={len(track_map):,} artists={len(artist_map):,}")
    print(f"train={train.height:,} val={val.height:,} test={test.height:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "download", "extract", "process"])
    args = parser.parse_args()

    if args.cmd in ("prepare", "download"):
        archive = download()
    if args.cmd in ("prepare", "extract"):
        archive = RAW_DIR / "lastfm-dataset-1K.tar.gz"
        tsv = extract(archive)
    if args.cmd in ("prepare", "process"):
        tsv = RAW_DIR / TSV_NAME
        process(tsv)


if __name__ == "__main__":
    main()
