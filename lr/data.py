"""Loading the scored panel, auditing its provenance, and the leakage-free sigma rescale."""

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DETECTORS", "zcols", "Z", "SCORED_COLUMNS", "load_scored_manifest",
           "provenance", "sigma_rescale", "family_registry", "generator_registry", "kfolds"]

DETECTORS = ["cnndetection", "univfd",
             "dmimagedetection_progan", "dmimagedetection_latent",
             "aeroblade"]
zcols = {d: d.replace("dmimagedetection_", "dmid_") for d in DETECTORS}
Z = list(zcols.values())

# The seven columns the detectors actually consumed, in the order build_manifest.py wrote them.
SCORED_COLUMNS = ["image_id", "path", "label", "generator_family",
                  "release_date", "generator", "crop_path"]

EXPECTED_SPLIT = {"authentic": (306, 306), "autoregressive": (8, 9), "diffusion": (137, 135),
                  "gan": (111, 110), "graphics": (8, 9), "inpainting": (26, 25), "other": (16, 18)}


def load_scored_manifest(root="."):
    """manifest.csv joined to the five cached detector score CSVs and the train/val split.

    The split lives only in data/{authentic,fakes}/*_train.parquet, never in manifest.csv;
    common.assign_splits stratified it by generator precisely so a per-family score model can
    be calibrated, so it is used as the importers intended rather than re-drawn.

    COCO2017_train and COCO2017_val are merged into one COCO2017 source (n=204, matching the
    README); crop_confound.ipynb section 4 keyed on COCO2017_train alone and dropped 8 images.
    """
    root = Path(root)
    sys.path.insert(0, str(root / "imports" / "sample"))
    import common

    man = pd.read_csv(root / "manifest.csv")
    M = man.copy()
    for d in DETECTORS:
        s = pd.read_csv(root / "scores" / f"{d}.csv")
        assert list(s.columns) == ["image_id", "raw_score"], f"{d}: unexpected columns"
        M = M.merge(s.rename(columns={"raw_score": d}), on="image_id", how="left",
                    validate="one_to_one")
    assert len(M) == len(man) == 1224, "merge changed the row count"
    for d in DETECTORS:
        assert M[d].notna().all(), f"{d}: {M[d].isna().sum()} images unscored"
        assert M[d].std() > 1e-6, f"{d}: score distribution collapsed"

    sample = common.load_sample()
    M = M.merge(sample[["file_id", "split"]].rename(columns={"file_id": "image_id"}),
                on="image_id", how="left", validate="one_to_one")
    assert M["split"].notna().all(), f"{M['split'].isna().sum()} images have no split"

    M["release_date"] = pd.to_datetime(M.release_date, errors="coerce")
    M["source"] = M.image_id.str.split("/").str[0].where(M.label == 0, "fake")
    M.loc[M.source.isin(["COCO2017_train", "COCO2017_val"]), "source"] = "COCO2017"
    M["frac"] = (200 * 200) / (M.source_width * M.source_height)   # the covariate
    M["logfrac"] = np.log10(M.frac)

    got = M.pivot_table(index="generator_family", columns="split", values="image_id",
                        aggfunc="size")
    for fam, (n_tr, n_va) in EXPECTED_SPLIT.items():
        assert (got.loc[fam, "train"], got.loc[fam, "val"]) == (n_tr, n_va), \
            f"{fam}: split is {tuple(got.loc[fam])}, expected {(n_tr, n_va)}"
    return M


def provenance(root="."):
    """Audit the sidecars' manifest hash against the file as it stands now.

    The recorded hash no longer matches, because commit ec1b4f6 APPENDED six crop-geometry
    columns after the scores were computed. Re-serialising just the seven scored columns
    reproduces the recorded hash exactly, which is the claim that has to hold for the cached
    scores to be usable. Returns (live, rebuilt, recorded, sidecar_table).
    """
    root = Path(root)
    live = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()
    as_str = pd.read_csv(root / "manifest.csv", dtype=str, keep_default_na=False)
    rebuilt = hashlib.sha256(
        as_str[SCORED_COLUMNS].to_csv(index=False, lineterminator="\n").encode()).hexdigest()

    meta = [json.load(open(root / "scores" / f"{d}.meta.json")) for d in DETECTORS]
    recorded = {m["detector"]: m["manifest"]["sha256"] for m in meta}
    assert len(set(recorded.values())) == 1, "the detectors disagree on which manifest they scored"
    scored = next(iter(set(recorded.values())))
    assert rebuilt == scored, "the scored columns have changed since the panel ran -- rescore"

    table = pd.DataFrame(meta)[["detector", "score_semantics", "crop_policy", "elapsed_s"]]
    assert table.crop_policy.nunique() == 1, "detectors saw different crops"
    return live, rebuilt, scored, table


def sigma_rescale(M):
    """Add sigma-above-authentic columns, reference fitted on authentic TRAIN rows only.

    merge_scores.ipynb section 4 and temporal_correlation.ipynb section 1 both reference all 612
    authentics, which is fine for descriptive plots but leaks the validation half into the
    reference a calibrator is fitted against. Returns (M, max |z_train - z_all| per detector).
    """
    M = M.copy()
    ref = M[(M.label == 0) & (M.split == "train")]
    deltas = {}
    for d in DETECTORS:
        all_auth = M.loc[M.label == 0, d]
        z_train = (M[d] - ref[d].mean()) / ref[d].std()
        z_all = (M[d] - all_auth.mean()) / all_auth.std()
        M[zcols[d]] = z_train
        deltas[zcols[d]] = float(np.abs(z_train - z_all).max())
    return M, deltas


def family_registry(M):
    """Families ordered by availability date = earliest release among their generators."""
    return (M[M.label == 1].groupby("generator_family")
            .agg(n=("image_id", "size"), ngen=("generator", "nunique"),
                 date=("release_date", "min"))
            .sort_values("date"))


def generator_registry(M):
    """The 36 generators ordered by release date, with their family."""
    return (M[M.label == 1].groupby("generator")
            .agg(fam=("generator_family", "first"), date=("release_date", "min"))
            .sort_values("date"))


def kfolds(df, k, stratum_col, seed=0):
    """k folds, stratified, each stratum seeded from its own name.

    assign_splits' idiom, so gaining or losing a stratum leaves the others' assignment
    untouched. df must have a fresh RangeIndex.
    """
    fold = np.full(len(df), -1)
    for stratum, idx in df.groupby(stratum_col, observed=True).indices.items():
        members = list(idx)
        random.Random(f"{seed}:{stratum}").shuffle(members)
        for pos, i in enumerate(members):
            fold[i] = pos % k
    assert (fold >= 0).all()
    return fold
