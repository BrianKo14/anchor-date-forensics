"""The synthetic half: 72 parquet shards from HuggingFace -> 180,000 q95 JPEGs.

    lrzpellegrini/AI-GenBench-fake_part
    train       57 shards, 144,000 rows, 4,000 per generator
    validation  15 shards,  36,000 rows, 1,000 per generator
    total       ~35.2 GB, no auth, no gating

**Why this does not use `huggingface_hub.snapshot_download`.** `resume_download` was removed in
huggingface_hub 1.0, and `_download_to_tmp_and_move` now writes to a process-unique
`<etag>.<uuid8>.incomplete` whose `finally` block unlinks it -- the source comment reads *"do not
keep a partial file around: it could not be reused anyway since the temporary name is unique to this
download"*. With the default `max_workers=8`, one dropped SSH session throws away up to 8 in-flight
shards, about 4 GB. requirements.txt pins `huggingface_hub>=1.29`, so that is the behaviour we would
get. Since the shards are plain files behind a `resolve/` URL that honours `Range`, fetching them
with `fetch.download_artifact` gives real byte-level resume that the library no longer offers.

Extraction is per row group, never per row: `image.bytes` is essentially the whole file, so pulling
one image at a time would re-read hundreds of megabytes per image. The metadata columns are written
out to a sidecar per shard so `manifest.py` can rebuild the manifest without touching image bytes
again.
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

import config  # noqa: F401 -- must precede common: it sets AIGENBENCH_DATA_ROOT
import fetch
from common import FAKES_DIR, image_path
from imaging import prepare_image
from sources import cpu_workers

NAME = "fakes"

META_COLUMNS = ["file_id", "generator", "origin_dataset", "description", "width", "height"]
META_DIR = config.CACHE_DIR / "fake_meta"

# The shards are the canonical published form of the fake half, and 35 GB against 1.5 TB free is
# cheap insurance against ever having to re-pull them. Set to 0 to reclaim the space.
KEEP_PARQUET = os.environ.get("AIGENBENCH_KEEP_PARQUET", "1") != "0"


def shard_name(split, index):
    return f"{split}-{index:05d}-of-{config.HF_SHARDS[split]:05d}.parquet"


def shard_url(split, index):
    return (f"https://huggingface.co/datasets/{config.HF_REPO}/resolve/main/"
            f"data/{shard_name(split, index)}")


def plan(ctx):
    total = 0
    for split in config.SPLITS:
        count = config.HF_SHARDS[split]
        if ctx.smoke:
            count = 1  # one shard already holds every generator
        for index in range(count):
            name = shard_name(split, index)
            ctx.state.add_artifact(
                name=f"fakes:{name}",
                source=NAME,
                url=shard_url(split, index),
                dest=config.PARQUET_CACHE / name,
                expected_bytes=None,  # resolved by HEAD at download time
            )
            total += 1
    if ctx.smoke:
        # One shard per split, ~2,500 rows each. Reporting the full 180,000 here would leave the
        # dashboard showing 3% complete at the end of a successful rehearsal.
        expected = total * 2500
    else:
        expected = sum(config.FAKE_ROWS_PER_GENERATOR[s] * config.N_GENERATORS
                       for s in config.SPLITS)
    print(f"fakes: {total} shards planned, ~{expected:,} images expected")
    return expected


def fetch_shard(ctx, record):
    """Download one shard with byte-range resume. Returns True when the file is on disk."""
    dest = record["dest"]
    expected = record["expected_bytes"]
    if expected is None:
        _, expected, _ = fetch.head(record["url"])
        if expected:
            ctx.state.update_artifact(record["name"], expected_bytes=expected)

    good, size, reason = fetch.download_artifact(
        record["url"], dest, expected_bytes=expected, bucket=ctx.bucket, stop=ctx.stop,
        on_progress=ctx.progress(NAME), connections=max(1, ctx.concurrency(NAME)),
        gate=ctx.gate(NAME),
    )
    if not good:
        # Same reasoning as coco.fetch_artifacts: an interrupt leaves a resumable .part, not a
        # broken shard, and must not be reported as a failure.
        interrupted = reason == "interrupted"
        ctx.state.update_artifact(record["name"], status="pending" if interrupted else "failed",
                                  error=None if interrupted else reason, got_bytes=size)
        return False
    ctx.state.update_artifact(record["name"], status="fetched", got_bytes=size)
    return True


def extract_shard(ctx, record, split):
    """Write every image in a shard, plus a metadata sidecar. Returns (written, skipped)."""
    import pandas as pd

    path = record["dest"]
    parquet = pq.ParquetFile(path)
    written = skipped = 0
    meta_rows = []

    for group in range(parquet.metadata.num_row_groups):
        if ctx.stop.is_set():
            return written, skipped
        if not ctx.wait_while_paused(NAME):
            return written, skipped

        table = parquet.read_row_group(group, columns=["image", *META_COLUMNS])
        images = table.column("image").to_pylist()
        columns = {name: table.column(name).to_pylist() for name in META_COLUMNS}

        def convert(position):
            file_id = columns["file_id"][position]
            dest = image_path(FAKES_DIR, file_id)
            record_meta = {name: columns[name][position] for name in META_COLUMNS}
            if dest.exists():
                record_meta["source_format"] = None  # normalised on an earlier run
                record_meta["split"] = split
                return record_meta, 0
            raw = images[position]["bytes"]
            with Image.open(io.BytesIO(raw)) as image:
                record_meta["source_format"] = image.format
                data = prepare_image(image)
            fetch.write_atomic(dest, data)
            record_meta["split"] = split
            return record_meta, len(data)

        workers = max(1, cpu_workers(ctx))
        group_items, group_bytes = [], 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fakes") as pool:
            for meta, size in pool.map(convert, range(len(images))):
                meta_rows.append(meta)
                group_items.append({
                    "source": NAME,
                    "split": split,
                    "file_id": meta["file_id"],
                    "dest": f"fakes/images/{meta['file_id'].replace('/', '_')}.jpg",
                    "bytes": size,
                })
                if size:
                    written += 1
                    group_bytes += size
                else:
                    skipped += 1

        # One transaction per row group rather than per image: ~2,500 rows at a time instead of
        # 180,000 separate commits over the run.
        ctx.state.add_done_items(group_items)
        if group_bytes:
            ctx.state.record_bytes(NAME, group_bytes)

    META_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(meta_rows)
    # Sidecar carries the source container per image, which cannot be recovered from disk once the
    # file has been normalised -- the same problem imports/sample/fakes.py solves by carrying the
    # previous manifest's value forward.
    frame.to_parquet(META_DIR / (path.name.replace(".parquet", "") + ".meta.parquet"), index=False)
    return written, skipped


def run(ctx):
    done = failed = 0

    for split in config.SPLITS:
        count = 1 if ctx.smoke else config.HF_SHARDS[split]
        for index in range(count):
            if ctx.stop.is_set():
                return done, failed
            if not ctx.wait_while_paused(NAME):
                return done, failed
            ctx.sync_rate()

            name = f"fakes:{shard_name(split, index)}"
            record = ctx.state.artifact(name)
            if record["status"] == "done":
                continue

            record["dest"] = config.PARQUET_CACHE / shard_name(split, index)
            record["name"] = name

            if record["status"] != "fetched" or not record["dest"].exists():
                started = time.time()
                if not fetch_shard(ctx, record):
                    failed += 1
                    ctx.log("error", f"{shard_name(split, index)} download failed", NAME)
                    continue
                size = record["dest"].stat().st_size
                ctx.log("info", f"{shard_name(split, index)} fetched "
                                f"({size / 1e6:.0f} MB in {time.time() - started:.0f}s)", NAME)

            written, skipped = extract_shard(ctx, record, split)
            if ctx.stop.is_set():
                return done, failed

            ctx.state.update_artifact(name, status="done")
            done += written
            ctx.log("info", f"{shard_name(split, index)} extracted "
                            f"{written} images ({skipped} already on disk)", NAME)

            if not KEEP_PARQUET:
                record["dest"].unlink(missing_ok=True)

    return done, failed
