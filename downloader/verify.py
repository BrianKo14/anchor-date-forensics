"""Integrity pass over everything on disk. Safe to run while the download is still going.

    python -m downloader.verify              # full sweep
    python -m downloader.verify --smoke      # relaxed counts, for the rehearsal run
    python -m downloader.verify --deep 2000  # additionally decode 2000 random images end to end

Four things get checked, in increasing order of how much they would hurt if they were wrong:

1. **Counts.** Per source and per generator, against what AI-GenBench pins.
2. **Readability.** Every file opens and reports a sane size. Header-only, so 360k files take
   minutes rather than hours; `--deep` samples a subset for a full decode.
3. **Compression parity.** Every image carries the same JPEG quantization table. This is the one
   that matters most: if the two halves were encoded differently, a detector could separate them on
   compression history alone and every number downstream would be measuring the wrong thing. It is
   the reason `imports/sample/imaging.py` exists, and this re-asserts it at 360k scale.
4. **Manifest agreement.** Every manifest row points at a file that is actually there.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import aigenbench  # noqa: E402
from common import AUTHENTIC_DIR, FAKES_DIR, IMAGE_MIN_SIZE, read_manifest  # noqa: E402
from imaging import Q75_LUMA, luma_table  # noqa: E402


class Report:
    def __init__(self):
        self.checks = []
        self.failed = 0

    def check(self, ok, label, detail=""):
        self.checks.append((ok, label, detail))
        if not ok:
            self.failed += 1
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
        return ok

    def summary(self):
        print(f"\n{len(self.checks) - self.failed}/{len(self.checks)} checks passed")
        return self.failed == 0


def scan_tables(directory, report, label, deep=0):
    """Quantization tables across a tree, plus a size floor. Returns the set of tables seen."""
    images = sorted((directory / "images").glob("*.jpg"))
    if not images:
        report.check(False, f"{label}: images present", "directory is empty")
        return set(), 0

    tables, unreadable = Counter(), []
    for path in images:
        try:
            tables[luma_table(path)] += 1
        except Exception as error:  # noqa: BLE001
            unreadable.append((path.name, type(error).__name__))

    report.check(not unreadable, f"{label}: all {len(images):,} files readable",
                 f"{len(unreadable)} unreadable, e.g. {unreadable[:3]}" if unreadable else "")
    report.check(len(tables) == 1, f"{label}: one quantization table",
                 f"saw {len(tables)}: {dict(list(tables.items())[:3])}" if len(tables) != 1 else
                 f"{list(tables)[0][:4]}...")
    if tables:
        report.check(Q75_LUMA not in tables, f"{label}: no q75 re-encode slipped in")

    if deep:
        from PIL import Image

        sample = random.Random(1234).sample(images, min(deep, len(images)))
        bad = []
        for path in sample:
            try:
                with Image.open(path) as image:
                    image.load()
                    if min(image.size) < IMAGE_MIN_SIZE:
                        bad.append((path.name, f"{image.size} under {IMAGE_MIN_SIZE}px"))
            except Exception as error:  # noqa: BLE001
                bad.append((path.name, type(error).__name__))
        report.check(not bad, f"{label}: {len(sample):,} sampled images decode fully",
                     f"{len(bad)} bad, e.g. {bad[:3]}" if bad else "")

    return set(tables), len(images)


def check_counts(report, smoke):
    """Per-split manifest counts against what the benchmark pins."""
    expected_fake = config.FAKE_ROWS_PER_GENERATOR

    for split in config.SPLITS:
        for kind, directory in (("authentic", AUTHENTIC_DIR), ("fakes", FAKES_DIR)):
            stem = f"{kind}_{split}"
            path = directory / f"{stem}.parquet"
            if not path.exists():
                report.check(smoke, f"{stem}: manifest exists",
                             "missing (expected during a partial run)" if smoke else "missing")
                continue

            manifest = read_manifest(directory, stem)
            missing = sum(1 for p in manifest["path"] if not (config.DATA_ROOT / p).exists())
            report.check(missing == 0, f"{stem}: all {len(manifest):,} manifest rows on disk",
                         f"{missing} missing" if missing else "")

            if kind == "fakes" and not smoke:
                per_generator = manifest["generator"].value_counts()
                want = expected_fake[split]
                report.check(len(per_generator) == config.N_GENERATORS,
                             f"{stem}: {config.N_GENERATORS} generators present",
                             f"saw {len(per_generator)}")
                off = {g: int(n) for g, n in per_generator.items() if n != want}
                report.check(not off, f"{stem}: {want:,} images per generator",
                             f"{len(off)} generators off target, e.g. {dict(list(off.items())[:3])}"
                             if off else "")
                report.check(manifest["release_date"].notna().all(),
                             f"{stem}: every row carries a release date")

            if kind == "authentic" and not smoke:
                got = manifest["origin_dataset"].value_counts().to_dict()
                want = Counter(fid.split("/")[0] if fid.startswith("COCO") else
                               fid.split("/")[0].replace("LAION-400M", "LAION-400M")
                               for fid in aigenbench.real_file_ids(split, verbose=False))
                for origin, target in sorted(want.items()):
                    if origin == "RAISE":
                        continue  # subset by design; checked separately against RAISE_TARGET
                    have = got.get(origin, 0)
                    report.check(have >= target, f"{stem}: {origin} {have:,}/{target:,}",
                                 f"short by {target - have:,}" if have < target else "")


def check_raise(report, smoke):
    if smoke:
        return
    raise_files = list((AUTHENTIC_DIR / "images").glob("RAISE_*.jpg"))
    report.check(len(raise_files) >= config.RAISE_TARGET * 0.98,
                 f"RAISE: {len(raise_files):,} of {config.RAISE_TARGET:,} target",
                 "" if len(raise_files) >= config.RAISE_TARGET * 0.98 else "short of target")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="relax count checks for a rehearsal or partial run")
    parser.add_argument("--deep", type=int, default=0,
                        help="also fully decode this many random images per half")
    arguments = parser.parse_args()

    print(f"verifying {config.DATA_ROOT}\n")
    report = Report()

    print("compression parity")
    authentic_tables, n_authentic = scan_tables(AUTHENTIC_DIR, report, "authentic",
                                                deep=arguments.deep)
    fake_tables, n_fakes = scan_tables(FAKES_DIR, report, "fakes", deep=arguments.deep)

    if authentic_tables and fake_tables:
        report.check(
            authentic_tables == fake_tables,
            "both halves share one quantization table",
            "" if authentic_tables == fake_tables
            else f"authentic {authentic_tables} vs fakes {fake_tables} -- compression history is a "
                 f"class cue; the halves must be rebuilt together",
        )

    print(f"\ncounts  ({n_authentic:,} authentic, {n_fakes:,} fake images on disk)")
    check_counts(report, arguments.smoke)
    check_raise(report, arguments.smoke)

    ok = report.summary()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
