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

def load_scored_manifest(root="."):
    """manifest.csv joined to the five cached detector score CSVs and the train/val split.

    Works against either layout, because two exist and they differ in more than size:

      * The in-repo 612+612 sample. manifest.csv carries neither `split` nor `origin_dataset`,
        so the split is joined from data/{authentic,fakes}/*_train.parquet, where
        common.assign_splits put an internal 50/50 cut stratified by generator.

      * An experiments/<name>/ directory built from the full dataset. build_manifest.py now
        writes `split` and `origin_dataset` as first-class columns, and there `split` is
        AI-GenBench's OWN benchmark partition -- roughly 80/20, not 50/50, and not a cut we
        drew. Prefer it: it is the partition the benchmark was designed around, and re-drawing
        one would discard that.

    The two vocabularies are reconciled here ("validation" -> "val") so that everything
    downstream can keep saying `split == "val"` without caring which layout it was handed.

    COCO2017_train and COCO2017_val are merged into one COCO2017 source; crop_confound.ipynb
    section 4 keyed on COCO2017_train alone and silently dropped the rest.
    """
    root = Path(root)

    man = pd.read_csv(root / "manifest.csv", low_memory=False)
    M = man.copy()
    for d in DETECTORS:
        s = pd.read_csv(root / "scores" / f"{d}.csv")
        assert list(s.columns) == ["image_id", "raw_score"], f"{d}: unexpected columns"
        M = M.merge(s.rename(columns={"raw_score": d}), on="image_id", how="left",
                    validate="one_to_one")
    # No fixed row count: the same code loads 1,224 and 355,638. What must hold is that the
    # join changed nothing and every image came back scored -- a short or duplicated join is
    # the failure that would otherwise reach calibration unnoticed.
    assert len(M) == len(man), f"merge changed the row count: {len(man)} -> {len(M)}"
    for d in DETECTORS:
        assert M[d].notna().all(), f"{d}: {M[d].isna().sum()} images unscored"
        assert M[d].std() > 1e-6, f"{d}: score distribution collapsed"

    if "split" not in M.columns:
        sys.path.insert(0, str(root / "imports" / "sample"))
        import common
        sample = common.load_sample()
        M = M.merge(sample[["file_id", "split"]].rename(columns={"file_id": "image_id"}),
                    on="image_id", how="left", validate="one_to_one")
    M["split"] = M["split"].replace({"validation": "val"})
    assert M["split"].notna().all(), f"{M['split'].isna().sum()} images have no split"
    assert set(M["split"].unique()) <= {"train", "val"}, \
        f"unexpected split labels: {sorted(set(M['split'].unique()))}"

    M["release_date"] = pd.to_datetime(M.release_date, errors="coerce")
    if "origin_dataset" in M.columns:
        M["source"] = M.origin_dataset.where(M.label == 0, "fake")
    else:
        M["source"] = M.image_id.str.split("/").str[0].where(M.label == 0, "fake")
    M.loc[M.source.isin(["COCO2017_train", "COCO2017_val"]), "source"] = "COCO2017"
    M["frac"] = (200 * 200) / (M.source_width * M.source_height)   # the covariate
    M["logfrac"] = np.log10(M.frac)

    # Structural, not numeric. Exact per-family counts pinned the 612+612 sample and broke on
    # anything else; what actually has to hold for a per-family score model to be fittable is
    # that no family lands entirely on one side of the split.
    got = M.pivot_table(index="generator_family", columns="split", values="image_id",
                        aggfunc="size").fillna(0).astype(int)
    for side in ("train", "val"):
        assert side in got.columns, f"no {side} rows at all"
    empty = got[(got["train"] == 0) | (got["val"] == 0)]
    assert empty.empty, f"families missing from one side of the split:\n{empty}"
    return M


def provenance(root="."):
    """Audit the sidecars' manifest hash against the file as it stands now.

    Two hashes are compared because two things have recorded one. An experiments/<name>/ run
    records the sha256 of the manifest file exactly as the panel read it, so `live` matches
    outright. The in-repo sample's sidecars predate that: commit ec1b4f6 APPENDED six
    crop-geometry columns after those scores were computed, so only a re-serialisation of the
    seven columns the detectors actually consumed reproduces the recorded hash.

    Either match is a pass, and which one matched is worth knowing, so both are returned.
    Neither matching means the manifest changed under the scores and they must be recomputed.
    Returns (live, rebuilt, recorded, sidecar_table).
    """
    root = Path(root)
    live = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()
    as_str = pd.read_csv(root / "manifest.csv", dtype=str, keep_default_na=False,
                         low_memory=False)
    rebuilt = hashlib.sha256(
        as_str[SCORED_COLUMNS].to_csv(index=False, lineterminator="\n").encode()).hexdigest()

    meta = [json.load(open(root / "scores" / f"{d}.meta.json")) for d in DETECTORS]
    recorded = {m["detector"]: m["manifest"]["sha256"] for m in meta}
    assert len(set(recorded.values())) == 1, "the detectors disagree on which manifest they scored"
    scored = next(iter(set(recorded.values())))
    assert scored in (live, rebuilt), (
        "the manifest does not match what the panel scored, whole-file or scored-columns -- "
        "these score files belong to a different manifest, so rescore rather than reuse them")

    columns = ["detector", "score_semantics", "crop_policy", "elapsed_s"]
    table = pd.DataFrame(meta)[columns]
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
