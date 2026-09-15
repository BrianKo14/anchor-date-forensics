"""Build the dataset manifests once the bytes are down.

Emits the same schema `imports/sample/` does, so `build_manifest.py`, `panel_io.load_manifest` and
the detector panel read the full 360k dataset exactly as they read the 612-image sample:

    authentic_train.parquet / .jsonl        authentic_validation.parquet / .jsonl
    fakes_train.parquet     / .jsonl        fakes_validation.parquet     / .jsonl

Two deliberate differences from the sample's manifests, both recorded in `manifest_meta.json`:

**`split` carries the benchmark's own partition.** In the sample, every row came from AI-GenBench's
`train` split and `assign_splits` cut it 50/50 into an internal train/val. Here we have the
benchmark's real train/validation partition, which is the one the Next-Period evaluation is defined
against, so there is nothing to invent -- `split` is simply `train` or `validation`.

**`path` is relative to the data root, not the repo root.** The sample lives inside the repo, so
`PROJECT_ROOT`-relative paths worked. 70 GB does not live inside a git repo, so paths are relative
to `AIGENBENCH_DATA_ROOT` and the sidecar records which root that was.
"""

import json

import config  # noqa: F401 -- must precede common/aigenbench: it sets AIGENBENCH_DATA_ROOT
import aigenbench
from common import AUTHENTIC_DIR, FAKES_DIR, FAKE_LABEL, MANIFEST_COLUMNS, REAL_LABEL, write_manifest

FAKE_COLUMNS = [*MANIFEST_COLUMNS, "source_format", "release_date"]

ORIGIN_OF = {
    "coco": None,          # taken from the file_id prefix: COCO2017_train / COCO2017_val
    "laion": "LAION-400M",
    "raise": "RAISE",
}


def _dimensions(path):
    """(width, height) without decoding the image -- PIL reads only the header for .size."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001
        return (0, 0)


def _laion_descriptions():
    """{file_id: description} for the LAION rows, from the benchmark's shipped filelist."""
    out = {}
    for split in config.SPLITS:
        for entry in aigenbench.laion_filelist(split):
            out[f"LAION-400M/{entry['id']}"] = entry.get("description", "")
    return out


def build_authentic(ctx, verbose=True):
    """One manifest per benchmark split from the completed COCO / LAION / RAISE items."""
    descriptions = _laion_descriptions()
    written = {}

    for split in config.SPLITS:
        rows = []
        for source in ("coco", "laion", "raise"):
            for item in ctx.state.items_for_manifest(source=source, split=split):
                path = config.DATA_ROOT / item["dest"]
                if not path.exists():
                    continue  # reconcile will requeue it; it simply is not in this manifest
                width, height = _dimensions(path)
                file_id = item["file_id"]
                rows.append({
                    "file_id": file_id,
                    "origin_dataset": ORIGIN_OF[source] or file_id.split("/")[0],
                    "label": REAL_LABEL,
                    "generator": "",
                    "description": descriptions.get(file_id, ""),
                    "width": width,
                    "height": height,
                    "path": item["dest"],
                    "split": split,
                })

        rows.sort(key=lambda row: row["file_id"])
        manifest = write_manifest(rows, AUTHENTIC_DIR, f"authentic_{split}")
        written[split] = len(manifest)
        if verbose and len(manifest):
            print(f"authentic_{split}: {len(manifest):,} rows")
            print(manifest["origin_dataset"].value_counts().to_string())
    return written


def build_fakes(verbose=True):
    """One manifest per benchmark split, assembled from the per-shard metadata sidecars.

    The sidecars exist because a normalised JPEG on disk no longer reveals the container it arrived
    in, and `source_format` is the column that proves the fake half was not silently re-encoded
    somewhere upstream.
    """
    import pandas as pd

    from sources.fakes_hf import META_DIR

    generators = aigenbench.benchmark_generators()
    written = {}

    sidecars = sorted(META_DIR.glob("*.meta.parquet")) if META_DIR.exists() else []
    if not sidecars:
        print("no fake metadata sidecars yet -- skipping the fake manifests")
        return written

    frame = pd.concat([pd.read_parquet(p) for p in sidecars], ignore_index=True)
    frame = frame.drop_duplicates(subset="file_id", keep="first")

    for split in config.SPLITS:
        part = frame[frame["split"] == split]
        rows = []
        for record in part.itertuples():
            dest = f"fakes/images/{record.file_id.replace('/', '_')}.jpg"
            if not (config.DATA_ROOT / dest).exists():
                continue
            rows.append({
                "file_id": record.file_id,
                "origin_dataset": record.origin_dataset,
                "label": FAKE_LABEL,
                "generator": record.generator,
                "description": record.description,
                "width": record.width,
                "height": record.height,
                "path": dest,
                "split": split,
                "source_format": record.source_format,
                "release_date": generators.get(record.generator),
            })

        # Sorted by release date then id, so the manifest reads in the order the G<=T construction
        # walks it.
        rows.sort(key=lambda row: (generators.get(row["generator"]) or "", row["file_id"]))
        manifest = write_manifest(rows, FAKES_DIR, f"fakes_{split}")
        written[split] = len(manifest)
        if verbose and len(manifest):
            counts = manifest["generator"].value_counts()
            print(f"fakes_{split}: {len(manifest):,} rows, {len(counts)} generators, "
                  f"{counts.min()}-{counts.max()} each")
    return written


def build(ctx, verbose=True):
    print("\n--- manifests ---")
    authentic = build_authentic(ctx, verbose=verbose)
    fakes = build_fakes(verbose=verbose)

    meta = {
        "data_root": str(config.DATA_ROOT),
        "path_relative_to": "data_root",
        "split_semantics": "AI-GenBench benchmark split (train/validation), not an internal cut",
        "jpeg_quality": 95,
        "raise_target": config.RAISE_TARGET,
        "authentic_rows": authentic,
        "fake_rows": fakes,
        "columns": {"authentic": MANIFEST_COLUMNS, "fakes": FAKE_COLUMNS},
    }
    (config.DATA_ROOT / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {config.DATA_ROOT / 'manifest_meta.json'}")
    return meta
