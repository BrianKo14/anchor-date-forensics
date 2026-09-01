"""Import the authentic (real) half of the sample -> data/authentic.

Three sources, per AI-GenBench's pinned file-id lists: COCO 2017, the LAION-400M subset used by
ELSA_D3, and RAISE. (The benchmark README also lists ImageNet, but the shipped lists contain none.)

Nothing here downloads an origin archive: COCO images are resolved individually by id and LAION by
its scraped URL, so the cost scales with the sample size rather than the dataset size.

    python imports/sample/authentic.py
"""

import random
from collections import Counter

import requests
from PIL import Image

import aigenbench
from common import (
    AUTHENTIC_DIR,
    BENCHMARK_SPLIT,
    PROJECT_ROOT,
    REAL_LABEL,
    assign_splits,
    disk_mb,
    image_path,
    write_manifest,
)
from imaging import decode_and_validate, prepare_image

N_PER_SOURCE = 68  # 68 COCO + 68 LAION + N_RAISE = 144, matching the fake half
N_RAISE = 8        # RAISE TIFFs are ~20 MB each, so sample far fewer
SEED = 1234

COCO_DIRS = {"COCO2017_train": "train2017", "COCO2017_val": "val2017"}

# Go through the S3 endpoint, not the images.cocodataset.org vanity host: that host is a CNAME onto
# the same bucket but serves an *.s3.amazonaws.com certificate, so HTTPS fails hostname validation.
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"

RAISE_CSV = PROJECT_ROOT / "RAISE_urls.csv"
RAISE_SOURCE = "http://loki.disi.unitn.it/RAISE/confirm.php?package=all"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "anchor-date-forensics/0.1 (thesis prototype)"


def fetch(url, timeout=20):
    """GET raw bytes. Returns (content, reason) -- content is None on failure.

    Failures are expected here (LAION link rot), so we return the reason rather than raising, and
    tally the reasons in `harvest`: a systematic breakage and ordinary rot both show up as missing
    images, and only the breakdown tells them apart.
    """
    try:
        response = SESSION.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content, "ok"
    except requests.HTTPError as error:
        return None, f"http {error.response.status_code}"
    except Exception as error:
        return None, type(error).__name__


def harvest(candidates, n, origin, url_of, description_of=lambda candidate: ""):
    """Walk candidates until n images survive fetch + validation. Writes JPEGs, returns manifest rows.

    `candidates` are (file_id, payload) pairs and must already be shuffled: we stop at the first n
    that work, so the oversampling that covers link rot only stays unbiased if the order is random.

    Images already on disk are reused rather than refetched, so raising n on a later run only pulls
    the shortfall. That matters most for RAISE, where a single TIFF is most of the sample's bytes.
    """
    rows, outcomes = [], Counter()

    def row(file_id, payload, width, height, path):
        return {
            "file_id": file_id,
            "origin_dataset": origin,
            "label": REAL_LABEL,
            "generator": "",
            "description": description_of(payload),
            "width": width,
            "height": height,
            "path": str(path.relative_to(PROJECT_ROOT)),
        }

    for file_id, payload in candidates:
        if len(rows) >= n:
            break
        path = image_path(AUTHENTIC_DIR, file_id)
        if path.exists():
            with Image.open(path) as cached:
                width, height = cached.size
            outcomes["cached"] += 1
            rows.append(row(file_id, payload, width, height, path))
            continue

        raw, reason = fetch(url_of(payload))
        if raw is None:
            outcomes[reason] += 1
            continue
        image = decode_and_validate(raw)
        if image is None:
            outcomes["undecodable or under 200px"] += 1
            continue

        outcomes["ok"] += 1
        path.write_bytes(prepare_image(image))
        rows.append(row(file_id, payload, image.width, image.height, path))

    cached = outcomes["cached"]
    attempted = sum(outcomes.values()) - cached
    suffix = f" (+{cached} already on disk)" if cached else ""
    print(f"{origin:<16} kept {len(rows)}/{attempted} attempted{suffix}")
    for reason, count in outcomes.most_common():
        if reason not in ("ok", "cached"):
            print(f"  {reason:<28} {count}")
    return rows


def coco_url(file_id):
    prefix, image_id = file_id.split("/")
    return f"{COCO_BASE}/{COCO_DIRS[prefix]}/{int(image_id):012d}.jpg"


def harvest_coco(file_ids, rng):
    coco_ids = [fid for fid in file_ids if fid.startswith("COCO2017_")]
    rng.shuffle(coco_ids)
    return harvest(
        candidates=((fid, fid) for fid in coco_ids),
        n=N_PER_SOURCE,
        origin="COCO2017",
        url_of=coco_url,
    )


def harvest_laion(rng):
    entries = aigenbench.laion_filelist(BENCHMARK_SPLIT)
    print(f"{len(entries):,} LAION URLs in the {BENCHMARK_SPLIT} filelist")
    # The filelist already includes spare images beyond the ids used, precisely to absorb rot.
    rng.shuffle(entries)
    return harvest(
        candidates=((f"LAION-400M/{entry['id']}", entry) for entry in entries),
        n=N_PER_SOURCE,
        origin="LAION-400M",
        url_of=lambda entry: entry["url"],
        description_of=lambda entry: entry.get("description", ""),
    )


def harvest_raise(file_ids, rng):
    """RAISE is the one source with no open URL list -- the CSV sits behind a confirmation page."""
    if not RAISE_CSV.exists():
        print(f"{RAISE_CSV.name} not found -- skipping RAISE.")
        print(f"Get it from {RAISE_SOURCE} ('Get the images!')")
        return []

    import pandas as pd

    # file_ids look like "RAISE/r0a2e85f0t"; the CSV keys the same ids in its File column.
    wanted = {fid.split("/")[1]: fid for fid in file_ids if fid.startswith("RAISE/")}
    catalog = pd.read_csv(RAISE_CSV)
    matched = [(wanted[row.File], row.TIFF) for row in catalog.itertuples() if row.File in wanted]
    rng.shuffle(matched)
    print(f"{len(matched)} of {len(wanted)} {BENCHMARK_SPLIT}-split RAISE ids found in the CSV")

    return harvest(candidates=matched, n=N_RAISE, origin="RAISE", url_of=lambda url: url)


def main():
    (AUTHENTIC_DIR / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    file_ids = aigenbench.real_file_ids(BENCHMARK_SPLIT)

    rows = harvest_coco(file_ids, rng)
    rows += harvest_laion(rng)
    rows += harvest_raise(file_ids, rng)

    # Stratify by source: the three sources keep their proportions on both sides of the partition.
    assign_splits(rows, lambda row: row["origin_dataset"])

    manifest = write_manifest(rows, AUTHENTIC_DIR, f"authentic_{BENCHMARK_SPLIT}")
    print(f"\n{len(manifest)} authentic images, {disk_mb(AUTHENTIC_DIR):.1f} MB on disk")
    print(manifest["origin_dataset"].value_counts().to_string())
    print("splits:", manifest["split"].value_counts().to_dict())
    return manifest


if __name__ == "__main__":
    main()
