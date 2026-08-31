"""Import the synthetic (fake) half of the sample -> data/fakes.

Two images from each of the 36 benchmark generators, drawn from AI-GenBench's published fake part.

Getting at the original bytes is the whole difficulty. The validation split is ~7 GB across 15
shards, so streaming pulls far too much; and datasets-server's cached-assets endpoint, which is
cheap, re-encodes everything at JPEG quality 75 (see imaging.Q75_LUMA) -- that would bake a
compression generation into the fakes that the reals never went through.

So we read the parquet directly. It is columnar and `image.bytes` is ~100% of the file, which means
the metadata columns cost nothing to scan and only the row groups we actually draw from get
downloaded, at ~17 MB per 100 images.

    python imports/sample/fakes.py
"""

import io
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

import aigenbench
from common import (
    AUTHENTIC_DIR,
    FAKE_LABEL,
    FAKES_DIR,
    PROJECT_ROOT,
    SPLIT,
    disk_mb,
    image_path,
    read_manifest,
    write_manifest,
)
from imaging import check_parity, prepare_image

REPO = "lrzpellegrini/AI-GenBench-fake_part"
SHARD = "validation-00000-of-00015.parquet"  # any one shard holds all 36 generators

N_PER_GENERATOR = 2  # 2 x 36 generators = 72, matching the authentic half
MAX_ROW_GROUPS = 8   # guard: each row group is ~17 MB, so cap the worst case at ~136 MB

INDEX_COLUMNS = ["generator", "file_id", "width", "height", "origin_dataset", "description"]


def open_shard():
    filesystem = HfFileSystem()
    return pq.ParquetFile(filesystem.open(f"datasets/{REPO}/data/{SHARD}", "rb"))


def build_index(parquet, verbose=True):
    """Scan the metadata columns to get a row -> generator map, without touching image bytes."""
    metadata = parquet.metadata
    rows_per_group = metadata.row_group(0).num_rows
    mb_per_group = metadata.row_group(0).total_byte_size / 1e6

    if verbose:
        print(f"{metadata.num_rows} rows in {metadata.num_row_groups} row groups "
              f"of {rows_per_group}, ~{mb_per_group:.0f} MB each")

    index = parquet.read(columns=INDEX_COLUMNS).to_pandas()
    index["row_group"] = index.index // rows_per_group

    if verbose:
        print(f"indexed {len(index)} rows, {index['generator'].nunique()} distinct generators")
    return index, rows_per_group, mb_per_group


def select_row_groups(index, generators, n_row_groups, mb_per_group, verbose=True):
    """Fewest row groups that supply N_PER_GENERATOR of every generator.

    Rows are shuffled with respect to generator rather than grouped by it, so a single 100-row group
    already covers most of the 36 and a handful covers all of them.
    """
    selected, available = [], Counter()
    for group in range(n_row_groups):
        if all(available[name] >= N_PER_GENERATOR for name in generators):
            break
        if len(selected) == MAX_ROW_GROUPS:
            break
        selected.append(group)
        available.update(index.loc[index["row_group"] == group, "generator"])

    short = {name: available[name] for name in generators if available[name] < N_PER_GENERATOR}
    if short:
        raise RuntimeError(
            f"still short of {N_PER_GENERATOR} after {len(selected)} row groups "
            f"(cap {MAX_ROW_GROUPS}): {short}"
        )

    if verbose:
        print(f"row groups {selected} cover all {len(generators)} generators "
              f"-> ~{len(selected) * mb_per_group:.0f} MB if not already on disk")
    return selected


def pick_rows(index, selected, generators):
    """Deterministic: the first N_PER_GENERATOR rows of each generator, no seed needed."""
    return (
        index[index["row_group"].isin(selected) & index["generator"].isin(generators)]
        .groupby("generator", sort=False)
        .head(N_PER_GENERATOR)
    )


def download_images(parquet, selected, picked):
    """Fetch the chosen row groups and return {file_id: stored bytes}."""
    table = parquet.read_row_groups(selected, columns=["image", "file_id"])
    stored = {
        file_id: record["bytes"]
        for file_id, record in zip(table.column("file_id").to_pylist(),
                                   table.column("image").to_pylist())
    }
    return {record.file_id: stored[record.file_id] for record in picked.itertuples()}


def main():
    (FAKES_DIR / "images").mkdir(parents=True, exist_ok=True)
    generators = aigenbench.benchmark_generators()

    parquet = open_shard()
    index, _, mb_per_group = build_index(parquet)
    selected = select_row_groups(index, generators, parquet.metadata.num_row_groups, mb_per_group)
    picked = pick_rows(index, selected, generators)
    print(f"{len(picked)} rows picked across {picked['generator'].nunique()} generators")

    # The index scan is cheap but the row groups are not, so only pay for them if something is
    # actually missing. Lets the script be re-run to rebuild the manifest without re-downloading.
    missing = [r for r in picked.itertuples() if not image_path(FAKES_DIR, r.file_id).exists()]
    stored = download_images(parquet, selected, picked) if missing else {}
    if not missing:
        print("all images already on disk -- skipping the row-group download")

    # A cached image has already been normalized, so its original container is no longer readable
    # off disk. Carry the recorded value forward rather than dropping the column to nulls.
    known_formats = {}
    if (FAKES_DIR / f"fakes_{SPLIT}.parquet").exists():
        previous = read_manifest(FAKES_DIR, f"fakes_{SPLIT}")
        known_formats = dict(zip(previous["file_id"], previous["source_format"]))

    rows, source_formats = [], Counter()
    for record in picked.itertuples():
        path = image_path(FAKES_DIR, record.file_id)
        if record.file_id in stored:
            with Image.open(io.BytesIO(stored[record.file_id])) as image:
                source_format = image.format
                path.write_bytes(prepare_image(image))
        else:
            source_format = known_formats.get(record.file_id)  # normalized on an earlier run
        source_formats[source_format or "unknown"] += 1

        rows.append({
            "file_id": record.file_id,
            "origin_dataset": record.origin_dataset,
            "label": FAKE_LABEL,
            "generator": record.generator,
            "description": record.description,
            "width": record.width,
            "height": record.height,
            "source_format": source_format,
            "path": str(path.relative_to(PROJECT_ROOT)),
        })

    rows.sort(key=lambda r: (generators[r["generator"]], r["file_id"]))
    for row in rows:
        row["release_date"] = generators[row["generator"]]

    manifest = write_manifest(rows, FAKES_DIR, f"fakes_{SPLIT}")
    print(f"\n{len(manifest)} fake images, {disk_mb(FAKES_DIR):.1f} MB on disk")
    per_generator = manifest["generator"].value_counts()
    print(f"{len(per_generator)} generators, {per_generator.min()}-{per_generator.max()} each")
    print("source containers:",
          ", ".join(f"{fmt} {n}" for fmt, n in source_formats.most_common()))

    print()
    check_parity(AUTHENTIC_DIR, FAKES_DIR)
    return manifest


if __name__ == "__main__":
    main()
