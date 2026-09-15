"""Access to the cloned AI-GenBench repository and the resources it ships.

The benchmark pins its exact image selection in file-id lists checked into the repo, so the repo is
the source of truth for *which* images belong in the sample. Cloning it is the cheapest way to read
those lists.
"""

import json
import subprocess
import sys
import zipfile
from collections import Counter

from common import AIGENBENCH_DIR

REPO_URL = "https://github.com/MI-BioLab/AI-GenBench.git"
RESOURCES = AIGENBENCH_DIR / "dataset_creation" / "resources"
METADATA_DIR = AIGENBENCH_DIR / "training_and_evaluation"


def ensure_clone():
    """Shallow-clone the benchmark repo if it isn't already here."""
    if AIGENBENCH_DIR.exists():
        return AIGENBENCH_DIR
    print(f"cloning {REPO_URL} -> {AIGENBENCH_DIR}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(AIGENBENCH_DIR)],
        check=True,
    )
    return AIGENBENCH_DIR


def real_file_ids(split, verbose=True):
    """The benchmark's pinned list of authentic image ids for a split, e.g. 'COCO2017_train/13992'."""
    ensure_clone()
    file_ids = (RESOURCES / f"{split}_real_file_ids.txt").read_text().split()

    if verbose:
        by_source = Counter(fid.split("/")[0] for fid in file_ids)
        print(f"{split} split: {len(file_ids):,} authentic images")
        for source, n in by_source.most_common():
            print(f"  {source:<16} {n:>7,}  ({n / len(file_ids):.1%})")
    return file_ids


def laion_filelist(split):
    """The scraped LAION URLs for a split. Ships zipped; extract on first use."""
    ensure_clone()
    path = RESOURCES / f"{split}_laion400m_filelist.json"
    if not path.exists():
        with zipfile.ZipFile(path.with_suffix(".zip")) as archive:
            archive.extractall(RESOURCES)
    return json.loads(path.read_text())


def benchmark_generators():
    """The generator registry: {name: release date}, 36 entries from 2017-03 to 2024-08.

    This is the spine of the thesis's G<=T construction -- the set of generators available by a given
    date -- so it is read from the benchmark rather than duplicated here.
    """
    ensure_clone()
    if str(METADATA_DIR) not in sys.path:
        sys.path.insert(0, str(METADATA_DIR))
    from ai_gen_bench_metadata.benchmark_generators import BENCHMARK_GENERATORS

    return BENCHMARK_GENERATORS
