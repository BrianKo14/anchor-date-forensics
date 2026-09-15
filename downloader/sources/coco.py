"""COCO 2017: two zips, unpacked locally into q95 JPEGs.

95,158 of the 180,000 pinned real images come from COCO, and the benchmark needs 91,294 of
train2017's 118,287 -- 77% of it. At that ratio, fetching images one at a time is the wrong trade:
95,158 individual requests against an S3 bucket at ~1.9 MB/s, each with its own round trip, versus
two range-resumable archives totalling 18.8 GiB. The archives win on wall clock and have two failure
modes instead of ninety-five thousand.

    val2017.zip     815,585,330 bytes   5,000 images
    train2017.zip 19,336,861,798 bytes 118,287 images

Both verified to answer `Range` with HTTP 206, so a killed download resumes instead of restarting.

The zips are deleted once every needed member has been extracted (see `cleanup`), because they are
trivially re-fetchable and 18.8 GiB is real money on a shared disk.

Descriptions are left empty, matching imports/sample/authentic.py. The captions live in a separate
252 MB annotations zip and nothing downstream reads the column for COCO rows; pulling them would
make the full dataset and the 612-image sample disagree about what a COCO row looks like.
"""

import threading
import zipfile

import config  # noqa: F401 -- must precede common/aigenbench: it sets AIGENBENCH_DATA_ROOT
import aigenbench
import fetch
from common import AUTHENTIC_DIR, image_path
from imaging import decode_and_validate, prepare_image
from sources import cpu_workers, pump

NAME = "coco"

# Only the two image zips. The annotations zip is not fetched -- see the module docstring.
NEEDED_ZIPS = ("val2017", "train2017")

_zips = threading.local()


def needed_zips(ctx):
    """A smoke run takes val2017 only: 815 MB rehearses the identical code path that train2017
    does, and pulling 19 GB to prove a zip can be opened would defeat the point of a rehearsal."""
    return ("val2017",) if ctx.smoke else NEEDED_ZIPS


def wanted_prefixes(ctx):
    return ("COCO2017_val",) if ctx.smoke else tuple(config.COCO_DIRS)


def member_name(file_id):
    """'COCO2017_train/96923' -> 'train2017/000000000096923.jpg' as COCO names it (12 digits)."""
    prefix, image_id = file_id.split("/")
    folder = config.COCO_DIRS[prefix]
    return f"{folder}/{int(image_id):012d}.jpg"


def zip_for(file_id):
    """The open ZipFile holding this id, cached per thread.

    Per thread rather than shared: ZipFile is not thread-safe, and re-opening train2017.zip per
    image would re-parse a 118,287-entry central directory ninety thousand times.
    """
    prefix = file_id.split("/")[0]
    name = config.COCO_DIRS[prefix]
    if not hasattr(_zips, "handles"):
        _zips.handles = {}
    if name not in _zips.handles:
        path = config.ZIP_CACHE / f"{name}.zip"
        if not path.exists():
            raise SystemExit(f"{path} missing -- run the artifact phase first")
        _zips.handles[name] = zipfile.ZipFile(path)
    return _zips.handles[name]


def plan(ctx):
    for name in needed_zips(ctx):
        suffix, size, md5 = config.COCO_ZIPS[name]
        ctx.state.add_artifact(
            name=f"coco:{name}",
            source=NAME,
            url=f"{config.COCO_BASE}/{suffix}",
            dest=config.ZIP_CACHE / f"{name}.zip",
            expected_bytes=size,
            md5=md5,
        )

    prefixes = wanted_prefixes(ctx)
    rows = []
    for split in config.SPLITS:
        ids = [fid for fid in aigenbench.real_file_ids(split, verbose=False)
               if fid.split("/")[0] in prefixes]
        ids = ids[:ctx.target_count(len(ids))]
        for file_id in ids:
            rows.append({
                "source": NAME,
                "split": split,
                "file_id": file_id,
                "url": None,  # comes out of a local archive, not over the network
                "dest": str(image_path(AUTHENTIC_DIR, file_id).relative_to(config.DATA_ROOT)),
            })
    ctx.state.add_items(rows)
    print(f"COCO: {len(rows)} images pinned across {len(config.SPLITS)} splits")
    return len(rows)


def fetch_artifacts(ctx):
    """Pull the zips, resuming any partial download. Returns True when all are present."""
    ok = True
    for name in needed_zips(ctx):
        if ctx.stop.is_set():
            return False
        record = ctx.state.artifact(f"coco:{name}")
        if record["status"] == "done":
            continue
        if not ctx.wait_while_paused(NAME):
            return False

        segments = max(1, ctx.concurrency(NAME))
        ctx.log("info", f"fetching {name}.zip ({record['expected_bytes'] / 1e9:.1f} GB)"
                        f" over {segments} parallel ranges", NAME)
        good, size, reason = fetch.download_artifact(
            record["url"], record["dest"], expected_bytes=record["expected_bytes"],
            bucket=ctx.bucket, stop=ctx.stop, on_progress=ctx.progress(NAME),
            connections=segments, gate=ctx.gate(NAME),
        )
        if not good:
            # An interruption is not a failure: the .part file is intact and the next run resumes
            # from its byte offset. Recording it as failed would put a red line in the dashboard for
            # what is actually the system working as designed.
            interrupted = reason == "interrupted"
            ctx.state.update_artifact(f"coco:{name}", status="pending" if interrupted else "failed",
                                      error=None if interrupted else reason, got_bytes=size)
            ctx.log("info" if interrupted else "error",
                    f"{name}.zip {'interrupted -- will resume' if interrupted else f'failed: {reason}'}",
                    NAME)
            ok = False
            continue

        if record["md5"]:
            digest = fetch.md5_file(record["dest"])
            if digest != record["md5"]:
                ctx.state.update_artifact(f"coco:{name}", status="failed",
                                          error=f"md5 {digest} != {record['md5']}")
                ctx.log("error", f"{name}.zip md5 mismatch", NAME)
                ok = False
                continue

        ctx.state.update_artifact(f"coco:{name}", status="done", got_bytes=size)
        ctx.log("info", f"{name}.zip complete ({size / 1e9:.1f} GB, {reason})", NAME)
    return ok


def handle_one(ctx, item):
    dest = config.DATA_ROOT / item["dest"]
    if dest.exists():
        return "done", dest.stat().st_size, None, "already on disk"

    try:
        with zip_for(item["file_id"]).open(member_name(item["file_id"])) as member:
            raw = member.read()
    except KeyError:
        return "failed", 0, None, "not in zip"

    image = decode_and_validate(raw)
    if image is None:
        return "failed", 0, None, "undecodable or under 200px"

    data = prepare_image(image)
    sha = fetch.write_atomic(dest, data)
    return "done", len(data), sha, "ok"


def cleanup(ctx):
    """Drop the zips once nothing is left to extract. 18.8 GiB back, re-fetchable if ever needed."""
    if ctx.smoke or ctx.state.pending_exists(NAME):
        return  # a smoke run keeps val2017.zip, so the real run does not refetch it
    for name in NEEDED_ZIPS:
        path = config.ZIP_CACHE / f"{name}.zip"
        if path.exists():
            path.unlink()
            ctx.log("info", f"removed {name}.zip (extraction complete)", NAME)


def run(ctx):
    if not fetch_artifacts(ctx):
        return 0, 0
    # Extraction is CPU-bound, not network-bound, so it is sized by the load governor rather than by
    # the (deliberately tiny) COCO download concurrency.
    result = pump(ctx, NAME, handle_one, batch=256, workers_of=cpu_workers)
    if not ctx.stop.is_set():
        cleanup(ctx)
    return result
