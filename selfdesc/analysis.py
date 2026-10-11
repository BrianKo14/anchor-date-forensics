"""Loading and analysing saved self-descriptions. Used by self_descriptions.ipynb.

Nothing here computes a description; that is extract.py's job. Two views of the same vectors:

  raw              Phi as extracted (after per-feature scaling), all images.
  family-centred   each family's mean Phi subtracted from its own images, then one PCA fitted on
                   the pooled remainder of the *fake* families (DECISIONS.md D-007). Granularity g
                   = the number of leading components kept. Authentic images are centred on their
                   own mean and projected onto the same axes, never used to fit them.

Separability is measured three ways, all on the same embedding:
  * leave-one-out k-NN *balanced* accuracy (mean per-class recall; k=5, Euclidean; chance =
    1/levels -- authentic is half the sample, so plain accuracy would sit on a 50% majority
    baseline and hide everything), against a permutation null: labels shuffled
    (within family, for within-family questions) with the neighbour graph held fixed -- the
    graph does not depend on labels, so the null costs one argsort, not one fit per shuffle;
  * KMeans with k = number of label levels -> ARI and NMI against the labels;
  * Calinski-Harabasz (between- over within-group dispersion, scaled by degrees of freedom).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score)

__all__ = ["configs", "load", "scale", "family_centre", "centred_pca", "neighbours",
           "knn_accuracy", "separability", "stratified_separability", "size_bucket", "FAMILY_ORDER", "GRANULARITIES"]

FAMILY_ORDER = ["authentic", "gan", "diffusion", "inpainting", "other", "autoregressive",
                "graphics"]
GRANULARITIES = (1, 2, 3, 5, 10, 20, 50)
KNN_K = 5


def configs(exp_dir):
    """Assembled configs under exp_dir (meta.json is written last, so it marks completion)."""
    root = Path(exp_dir) / "descriptions"
    return sorted(p.parent.name for p in root.glob("*/meta.json"))


def load(exp_dir, config):
    """(index, X float64, diagnostics, meta) for one config, rows aligned and checked."""
    exp_dir = Path(exp_dir)
    index = pd.read_parquet(exp_dir / "images.parquet")
    cfg = exp_dir / "descriptions" / config
    npz = np.load(cfg / "descriptions.npz")
    X, ids = npz["X"].astype(np.float64), npz["image_id"]
    diag = pd.read_parquet(cfg / "diagnostics.parquet")
    meta = json.loads((cfg / "meta.json").read_text())
    assert len(index) == len(X) == len(diag) == meta["rows"], "row counts disagree"
    assert (index["image_id"].to_numpy() == ids).all(), "descriptions not in images.parquet order"
    assert (diag["image_id"].to_numpy() == ids).all(), "diagnostics not in descriptions order"
    assert np.isfinite(X).all()
    return index, X, diag, meta


def scale(X, how="robust"):
    """Per-feature scaling before any distance is taken.

    "robust" (default): subtract the median, divide by the IQR. Phi's coordinates live on very
    different scales, and near-collinear residual channels give some fits large cancelling
    coefficients (heavy tails, see the QC section), which a mean/SD standardisation would let
    dominate. "standard": mean/SD. "none": Phi as extracted, which is what the paper's K-means
    clustering uses.
    """
    if how == "none":
        return X.copy()
    if how == "standard":
        return (X - X.mean(0)) / X.std(0)
    if how == "robust":
        q1, med, q3 = np.percentile(X, [25, 50, 75], axis=0)
        return (X - med) / np.where(q3 - q1 > 0, q3 - q1, 1.0)
    raise ValueError(how)


def family_centre(Z, family):
    """Z with each family's mean row subtracted from that family's rows."""
    family = np.asarray(family)
    out = Z.copy()
    for f in np.unique(family):
        rows = family == f
        out[rows] -= Z[rows].mean(axis=0)
    return out


def centred_pca(Zc, fit_rows, n_components=50, seed=0):
    """PCA fitted on Zc[fit_rows] (the fake families), applied to every row."""
    n_components = min(n_components, int(fit_rows.sum()) - 1, Zc.shape[1])
    pca = PCA(n_components=n_components, random_state=seed).fit(Zc[fit_rows])
    return pca.transform(Zc), pca


def neighbours(Z, k=KNN_K):
    """Indices of each row's k nearest other rows (self excluded), Euclidean."""
    sq = (Z ** 2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def knn_accuracy(nbrs, labels):
    """Leave-one-out k-NN balanced accuracy for a fixed neighbour graph.

    Ties go to the nearest label. Balanced = mean over classes of per-class recall.
    """
    labels = np.asarray(labels)
    codes, inverse = np.unique(labels, return_inverse=True)
    votes = inverse[nbrs]                                         # (n, k)
    n_cls = len(codes)
    counts = np.zeros((len(labels), n_cls))
    np.add.at(counts, (np.arange(len(labels))[:, None], votes), 1)
    # tiebreak: a tiny bonus by rank so the nearest neighbour among tied labels wins
    rank_bonus = np.zeros_like(counts)
    for j in range(nbrs.shape[1] - 1, -1, -1):
        rank_bonus[np.arange(len(labels)), votes[:, j]] = (nbrs.shape[1] - j) * 1e-3
    pred = (counts + rank_bonus).argmax(1)
    hit = pred == inverse
    return float(np.mean([hit[inverse == c].mean() for c in range(n_cls)]))


def _shuffle(labels, groups, rng):
    labels = np.asarray(labels).copy()
    if groups is None:
        return rng.permutation(labels)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        labels[rows] = labels[rng.permutation(rows)]
    return labels


def separability(Z, labels, groups=None, n_perm=500, seed=0, kmeans=True):
    """All separability measures of `labels` on embedding Z, plus the k-NN permutation null.

    `groups`, if given, restricts shuffling to within each group (e.g. generators shuffled only
    inside their own family), so the null keeps whatever the groups already explain.
    Returns a dict; `p` is the one-sided permutation p-value (with the +1 correction).
    """
    labels = np.asarray(labels)
    nbrs = neighbours(Z)
    acc = knn_accuracy(nbrs, labels)
    rng = np.random.default_rng(seed)
    null = np.array([knn_accuracy(nbrs, _shuffle(labels, groups, rng)) for _ in range(n_perm)])
    levels = np.unique(labels)
    out = {"n": len(labels), "levels": len(levels), "knn_acc": acc, "chance": 1.0 / len(levels)}
    if n_perm:
        out.update(null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)),
                   p=float((1 + (null >= acc).sum()) / (1 + n_perm)))
    if len(levels) > 1:
        out["calinski_harabasz"] = float(calinski_harabasz_score(Z, labels))
        if kmeans:
            km = KMeans(n_clusters=len(levels), n_init=10, random_state=seed).fit_predict(Z)
            out["ari"] = float(adjusted_rand_score(labels, km))
            out["nmi"] = float(normalized_mutual_info_score(labels, km))
    return out


def stratified_separability(Z, labels, strata, n_perm=500, seed=0, min_per_label=5):
    """Separability of `labels` *inside* each stratum, pooled. The deconfounding test.

    Within each stratum (e.g. one family x one origin dataset), keep labels with at least
    `min_per_label` rows; skip strata left with fewer than two labels. Neighbours are searched
    only inside the stratum, so stratum identity cannot help. Labels are shuffled only inside
    strata for the null. Pooled accuracy = per-stratum balanced accuracy weighted by rows, against
    the same pooled statistic under simultaneous within-stratum shuffles.
    Returns (pooled dict, per-stratum DataFrame).
    """
    labels, strata = np.asarray(labels), np.asarray(strata)
    rng = np.random.default_rng(seed)
    cells = []
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        counts = pd.Series(labels[rows]).value_counts()
        keep = counts[counts >= min_per_label].index
        rows = rows[np.isin(labels[rows], keep)]
        if len(keep) < 2:
            continue
        k = min(KNN_K, len(rows) - 1)
        cells.append((s, rows, neighbours(Z[rows], k=k)))
    if not cells:
        return None, pd.DataFrame()

    def pooled(lab):
        accs = [knn_accuracy(nb, lab[rows]) for _, rows, nb in cells]
        w = np.array([len(rows) for _, rows, _ in cells])
        return float(np.average(accs, weights=w)), accs

    obs, per_cell = pooled(labels)
    null = np.array([pooled(_shuffle(labels, strata, rng))[0] for _ in range(n_perm)])
    chance = float(np.average([1 / len(np.unique(labels[rows])) for _, rows, _ in cells],
                              weights=[len(rows) for _, rows, _ in cells]))
    table = pd.DataFrame({"stratum": [s for s, _, _ in cells],
                          "n": [len(rows) for _, rows, _ in cells],
                          "labels": [len(np.unique(labels[rows])) for _, rows, _ in cells],
                          "knn_acc": per_cell})
    table["chance"] = 1 / table["labels"]
    out = {"strata": len(cells), "n": int(table.n.sum()), "chance": chance, "knn_acc": obs,
           "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
           "p": float((1 + (null >= obs).sum()) / (1 + n_perm))}
    return out, table


def size_bucket(index):
    """Native short side, bucketed: the size covariate in categorical form."""
    short = np.minimum(index["source_width"], index["source_height"])
    return pd.cut(short, [0, 256, 512, 1024, np.inf],
                  labels=["<=256", "257-512", "513-1024", ">1024"]).astype(str)
