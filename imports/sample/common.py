"""Shared paths, constants and manifest I/O for the sample import.

Both halves of the sample are written with the same schema and the same JPEG settings; the constants
that enforce that live here so neither importer can drift from the other.
"""

import json
import os
import random
from pathlib import Path

# parents[2] rather than Path.cwd(): the importers are scripts, and they should behave the same
# whether they are run from the repo root, from this directory, or from anywhere else.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep dataset downloads inside the repo rather than in ~/.cache. Gitignored.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

AIGENBENCH_DIR = PROJECT_ROOT / "AI-GenBench"
DATA_DIR = PROJECT_ROOT / "data"
AUTHENTIC_DIR = DATA_DIR / "authentic"
FAKES_DIR = DATA_DIR / "fakes"

BENCHMARK_SPLIT = "validation"  # "validation" (36k ids) or "train" (144k); only the id list differs

# The train/val partition *within* the sample -- a different axis from BENCHMARK_SPLIT above, and
# recorded per row in the manifests' `split` column.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
SPLIT_SEED = 1234

# AI-GenBench constants, from dataset_creation/dataset_utils/common_utils.py
IMAGE_MIN_SIZE = 200
JPEG_QUALITY = 95

REAL_LABEL = 0
FAKE_LABEL = 1

# Column order shared by both manifests. Fakes add `source_format` and `release_date` on top.
MANIFEST_COLUMNS = [
    "file_id",
    "origin_dataset",
    "label",
    "generator",
    "description",
    "width",
    "height",
    "path",
    "split",
]


def assign_splits(rows, stratum_of, seed=SPLIT_SEED):
    """Partition rows 50/50 into TRAIN_SPLIT / VAL_SPLIT, stratified. Sets row["split"] in place.

    `stratum_of` names the group a row must be balanced within: the generator for the fake half, the
    origin dataset for the real one. Stratifying by generator is the point of the whole exercise --
    a per-family score model cannot be calibrated on a family that landed entirely on one side.

    Each stratum is seeded from its own name rather than from one global RNG, so gaining or losing a
    stratum leaves every other one's assignment untouched -- the same stability that lets the
    importers grow the sample without reshuffling what is already on disk.
    """
    by_stratum = {}
    for row in rows:
        by_stratum.setdefault(stratum_of(row), []).append(row)

    # Odd strata alternate which side gets the spare image, so the halves stay balanced overall.
    # Sizes are even at the target counts, but LAION link rot can leave a source short.
    odd_to_train = True

    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda row: row["file_id"])
        random.Random(f"{seed}:{stratum}").shuffle(members)

        n_train = len(members) // 2
        if len(members) % 2 and odd_to_train:
            n_train += 1
        if len(members) % 2:
            odd_to_train = not odd_to_train

        for position, row in enumerate(members):
            row["split"] = TRAIN_SPLIT if position < n_train else VAL_SPLIT

    return rows


def image_path(out_dir, file_id):
    """Where a given file_id's normalized JPEG lives. file_ids contain slashes; flatten them."""
    return out_dir / "images" / (file_id.replace("/", "_") + ".jpg")


def write_manifest(rows, out_dir, stem):
    """Write rows as both parquet (for downstream use) and jsonl (so diffs stay readable)."""
    import pandas as pd

    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(out_dir / f"{stem}.parquet", index=False)
    with open(out_dir / f"{stem}.jsonl", "w") as handle:
        for row in manifest.to_dict("records"):
            handle.write(json.dumps(row) + "\n")

    return manifest


def read_manifest(out_dir, stem):
    import pandas as pd

    return pd.read_parquet(out_dir / f"{stem}.parquet")


def sample_exists():
    """True when both halves have been imported."""
    return (AUTHENTIC_DIR / f"authentic_{BENCHMARK_SPLIT}.parquet").exists() and (
        FAKES_DIR / f"fakes_{BENCHMARK_SPLIT}.parquet"
    ).exists()


def load_sample():
    """Both halves as one DataFrame, authentic (label 0) first.

    Carries the `split` column the importers wrote, i.e. the train/val partition.
    """
    import pandas as pd

    authentic = read_manifest(AUTHENTIC_DIR, f"authentic_{BENCHMARK_SPLIT}")
    fakes = read_manifest(FAKES_DIR, f"fakes_{BENCHMARK_SPLIT}")
    return pd.concat([authentic, fakes], ignore_index=True)


def disk_mb(out_dir):
    return sum(p.stat().st_size for p in (out_dir / "images").glob("*.jpg")) / 1e6
