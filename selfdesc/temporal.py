"""Does the distribution of self-descriptions shift over time within a family? (DECISIONS.md D-008)

    python -m selfdesc.temporal --exp experiments/selfdesc-sample --perms 1000

Writes experiments/<exp>/temporal/*.parquet + meta.json; self_descriptions_time.ipynb only loads them.

The unit of evidence is the **generator**. Release date is a property of a generator, not of an
image, so the only temporal claim worth making is one that transfers to a generator the model
has not seen. Every test therefore leaves whole generators out, and every null shuffles release
dates *among generators* (re-binning each time). Images within a generator are not independent
draws with respect to time, and treating them as such would make any generator-level difference
look like a temporal effect with thousands of degrees of freedom.

  T1  time-bin attribution of unseen generators. Equal-generator-count bins by release date.
      Leave one generator out, fit one GMM per bin on the rest (components = clusters, count by
      BIC), and score the held-out generator under each. Statistic: mean over generators of the
      log-likelihood ratio of its true bin vs the mean of the other bins (nats / image).
  T2  drift vs jumps. Held-out divergence between (generator x origin) cells, against |delta date|.
  T3  what the per-bin GMM components contain (descriptive).
  T4  how many components each generator's own GMM wants (descriptive).

Diagnostics for *why* T1 is null (DECISIONS.md D-011), each aimed at one explanation:

  L1  lineage attribution: T1's machinery with lineage classes (selfdesc/lineage.csv: the
      generator's output stage, or its codebase) in place of time bins. Null: lineage labels
      permuted among generators. Asks whether Phi groups generators by what they inherit.
  L2  lineage vs date, pairwise: does held-out divergence between two generators follow
      "different lineage" or "|delta date|", each controlling for the other (rank regression
      on T2's cell pairs, a separate permutation null for each term).
  T5  linear drift in generator means, in within-generator SDs (Mahalanobis), with a
      group-bootstrap upper bound and a simulation of how many release groups a drift of a
      given size would need. Asks whether the null is a power problem.
  Subsets (`subset=`): single origin datasets (does family x origin centring remove the
      signal?) and native-size bands (is the diffusion hint just resolution?).
  S1  family and generator separability (5-NN, R-001 style), for the pipeline arms
      (q95 vs native crops, luma vs RGB): did the input pipeline wipe the signal out?

Features are computed without any date information (scaling, centring, PCA are unsupervised), so
fitting them on all rows does not leak the quantity being tested.
"""

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from . import analysis as A
from . import core

__all__ = ["phi_blocks", "features", "release_groups", "time_bins", "fit_gmm_bic",
           "logo_attribution", "logo_time_attribution",
           "t1_test", "planted_power", "size_check", "cell_divergence", "t2_test",
           "bin_components", "generator_components", "load_lineage", "lineage_test",
           "lineage_vs_date", "drift_test", "drift_power", "separability_test"]

MIN_IMAGES = 5            # a generator needs this many images in the subset to count (T1)
MIN_CELL = 10             # a generator x origin cell needs this many for T2 (>= 5 per held-out half)
MIN_PER_COMPONENT = 5     # BIC never considers more components than n / 5
MAX_COMPONENTS = 4
REG_COVAR = 1e-3
# Generators released within this many days of each other form one *release group*: held out
# together and permuted as one unit (DECISIONS.md D-009). Without it, near-twins such as FLUX Dev /
# Schnell (one release, one day apart) sit in each other's training bin under the observed dates
# but almost never under the null, which inflates T1.
RELEASE_WINDOW_DAYS = 7
TIME_FAMILIES = ("diffusion", "gan")
LINEAGE_CSV = Path(__file__).resolve().parent / "lineage.csv"
LINEAGE_COLUMNS = ("stage", "codebase")      # stage is the primary lineage label (D-011)
SIZE_BANDS = ("short<=256", "short>=512")    # native short side, for the resolution arm


# --- features --------------------------------------------------------------------------------

def phi_blocks(X, K, B, planes=1):
    """Named column blocks of Phi.

    "mean" is phi-bar = mean over the K residual channels of phi_k. The channels are ~0.93
    correlated, so phi-bar is the well-conditioned shared filter and "deviations" (phi_k - phi-bar)
    hold the large cancelling coefficients the joint fit produces. phi-bar is split further by
    neighbourhood ring (Chebyshev distance of the offset from the centre pixel) and by offset
    parity (both coordinates even: the lattice a 2x upsampling stage acts on).

    `planes` = 3 for an RGB description (core.load_channels): every block is then taken per
    colour plane and concatenated, so phi-bar is three filters, one per plane.
    """
    n = len(X)
    n_feat = B * B - 1
    phi = X.reshape(n, planes, K, n_feat)
    mean = phi.mean(axis=2)                                   # (n, planes, n_feat)
    c = B // 2
    offsets = [(i - c, j - c) for i in range(B) for j in range(B) if (i, j) != (c, c)]
    ring = np.array([max(abs(a), abs(b)) for a, b in offsets])
    even = np.array([a % 2 == 0 and b % 2 == 0 for a, b in offsets])

    def flat(a):
        return a.reshape(n, -1)

    blocks = {"full": X, "mean": flat(mean), "deviations": flat(phi - mean[:, :, None]),
              "mean ring 1": flat(mean[..., ring == 1]), "mean ring 2": flat(mean[..., ring == 2]),
              "mean ring 3+": flat(mean[..., ring >= 3]),
              "mean even offsets": flat(mean[..., even]), "mean odd offsets": flat(mean[..., ~even])}
    return blocks


def _planes(meta):
    return core.CHANNELS[meta.get("channels", "luma")]


def features(Xb, index, centring="family x origin", g=10, scaling="robust"):
    """Scale -> subtract group means -> PCA fitted on the fakes -> first g components (all rows)."""
    fake = (index["label"] == 1).to_numpy()
    if centring == "family":
        groups = index["generator_family"].to_numpy()
    elif centring == "family x origin":
        groups = (index["generator_family"] + " | " + index["origin_dataset"]).to_numpy()
    else:
        raise ValueError(centring)
    Z = A.family_centre(A.scale(Xb, scaling), groups)
    P, _ = A.centred_pca(Z, fake, n_components=min(g, Z.shape[1]))
    return P[:, :g]


# --- bins and mixtures -----------------------------------------------------------------------

def release_groups(dates, window=RELEASE_WINDOW_DAYS):
    """{generator: group id}: single-linkage on release date with a `window`-day threshold."""
    order = dates.sort_values(kind="stable")
    out, gid, prev = {}, -1, None
    for gen, date in order.items():
        if prev is None or (date - prev).days > window:
            gid += 1
        out[gen] = gid
        prev = date
    return pd.Series(out)


def group_dates(dates, groups):
    """Each generator's date replaced by its release group's earliest date."""
    first = dates.groupby(groups).min()
    return pd.Series(first[groups[dates.index]].to_numpy(), index=dates.index)


def time_bins(dates, n_bins, groups):
    """{generator: bin} with an equal number of *release groups* per bin, in date order.

    `dates` (Series indexed by generator) must be group dates, so a group's members share one
    date and one bin. Binning counts groups, not generators, because the group is the unit that is
    held out and permuted (D-009): every bin then keeps >= floor(groups / n_bins) groups under any
    permutation, so holding one out never empties a bin.
    """
    group_date = dates.groupby(groups[dates.index]).first().sort_values(kind="stable")
    n = len(group_date)
    group_bin = {grp: min(n_bins - 1, rank * n_bins // n) for rank, grp in
                 enumerate(group_date.index)}
    return {g: group_bin[groups[g]] for g in dates.index}


def fit_gmm_bic(Y, seed=0, max_components=MAX_COMPONENTS):
    """Diagonal-covariance GMM, number of components chosen by BIC (at most n / MIN_PER_COMPONENT)."""
    best = None
    top = max(1, min(max_components, len(Y) // MIN_PER_COMPONENT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in range(1, top + 1):
            gm = GaussianMixture(k, covariance_type="diag", reg_covar=REG_COVAR, n_init=1,
                                 random_state=seed).fit(Y)
            bic = gm.bic(Y)
            if best is None or bic < best[0]:
                best = (bic, gm)
    return best[1]


def logo_attribution(P, gens, classes, groups, n_classes, dates=None, seed=0):
    """Leave-one-release-group-out attribution of generators to classes. One row per generator.

    `classes` maps generator -> class index in 0..n_classes-1 (a time bin for T1, a lineage for
    L1). Each release group is held out whole; one GMM per class is fitted on the remaining
    generators, and each held-out generator is scored on its own. Every class must span >= 2
    release groups, so holding one out never empties a class. Columns: generator, group, date,
    true_bin, ll_<b> (mean log-likelihood per image under class b's GMM), pred_bin, abs_err
    (meaningful for ordered classes only), llr (true class minus the mean of the others).
    """
    groups_per_class = pd.Series({b: groups[[g for g in classes if classes[g] == b]].nunique()
                                  for b in range(n_classes)})
    assert groups_per_class.min() >= 2, \
        f"every class needs >= 2 release groups so one can be held out: {groups_per_class.to_dict()}"
    row_class = np.array([classes[g] for g in gens])
    row_group = groups[gens].to_numpy()
    rows = []
    for grp in np.unique(row_group):
        held = row_group == grp
        models = [fit_gmm_bic(P[~held & (row_class == b)], seed) for b in range(n_classes)]
        for g in np.unique(gens[held]):
            mine = gens == g
            ll = np.array([m.score_samples(P[mine]).mean() for m in models])
            true = classes[g]
            pred = int(np.argmax(ll))
            rows.append(dict(generator=g, group=int(grp),
                             date=dates[g] if dates is not None else pd.NaT, true_bin=true,
                             pred_bin=pred, abs_err=abs(pred - true),
                             llr=float(ll[true] - np.delete(ll, true).mean()),
                             **{f"ll_{b}": float(v) for b, v in enumerate(ll)}))
    return pd.DataFrame(rows)


def logo_time_attribution(P, gens, dates, groups, n_bins, seed=0):
    """Leave-one-release-group-out time-bin attribution (T1). One row per generator.

    `dates` must already be group dates (every member of a group shares one), so bins never split
    a group. See logo_attribution for the columns.
    """
    return logo_attribution(P, gens, time_bins(dates, n_bins, groups), groups, n_bins, dates,
                            seed)


def _summary(table):
    """Averaged over release groups (the independent units), each group = mean of its members."""
    by_group = table.assign(hit=table.pred_bin == table.true_bin).groupby("group")
    return {"acc": float(by_group.hit.mean().mean()), "mae": float(by_group.abs_err.mean().mean()),
            "llr": float(by_group.llr.mean().mean())}


def origin_source(index):
    """The origin dataset without its split suffix: "Forensynths/test" -> "Forensynths"."""
    return index["origin_dataset"].str.split("/").str[0]


def native_short_side(index):
    return np.minimum(index["source_width"], index["source_height"]).to_numpy(dtype=float)


def subset_mask(index, subset=None):
    """Rows a spec's `subset` keeps. None / "all"; "origin=X" or "origin!=X" (X an origin source,
    split suffix ignored); "short<=N" or "short>=N" (native short side in pixels)."""
    if subset in (None, "all"):
        return np.ones(len(index), dtype=bool)
    for op in ("!=", "<=", ">=", "="):
        if op in subset:
            key, value = subset.split(op, 1)
            break
    else:
        raise ValueError(subset)
    if key == "origin":
        src = origin_source(index).to_numpy()
        return src != value if op == "!=" else src == value
    if key == "short":
        short = native_short_side(index)
        return short <= float(value) if op == "<=" else short >= float(value)
    raise ValueError(subset)


def _subset(index, family, subset=None):
    """Row mask for a family within `subset`, keeping generators with >= MIN_IMAGES rows there."""
    sel = (index["generator_family"] == family).to_numpy(copy=True) & subset_mask(index, subset)
    counts = index[sel].groupby("generator").size()
    keep = counts[counts >= MIN_IMAGES].index
    return sel & index["generator"].isin(keep).to_numpy()


def _not_testable(n_generators, n_groups, n_images, n_bins, perms, reason):
    """A T1 summary for a subset too small to test, so a sweep reports it instead of crashing."""
    return ({"n_generators": n_generators, "n_groups": n_groups, "n_images": n_images,
             "n_bins": n_bins, "chance_acc": 1 / n_bins, "perms": perms, "acc": np.nan,
             "mae": np.nan, "llr": np.nan, "null_acc_mean": np.nan, "null_acc_q95": np.nan,
             "p_acc": np.nan, "null_llr_mean": np.nan, "null_llr_q95": np.nan, "p_llr": np.nan,
             "null_mae_mean": np.nan, "p_mae": np.nan, "not_testable": reason},
            pd.DataFrame(), pd.DataFrame({"acc": [], "mae": [], "llr": []}))


def t1_test(P, index, family, n_bins=3, perms=1000, seed=0, subset=None, n_jobs=-1):
    """T1 with its generator-level permutation null. Returns (summary dict, per-generator, nulls).

    A subset with fewer than 2 * n_bins release groups cannot be tested (every bin needs two, so
    one can be held out): it returns NaNs and a `not_testable` reason instead of raising.
    """
    sel = _subset(index, family, subset)
    gens = index.loc[sel, "generator"].to_numpy()
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    dates = group_dates(raw, groups)
    if groups.nunique() < 2 * n_bins:
        return _not_testable(len(dates), int(groups.nunique()), int(sel.sum()), n_bins, perms,
                             f"{groups.nunique()} release group(s) < 2 x {n_bins} bins")
    per_gen = logo_time_attribution(P[sel], gens, dates, groups, n_bins, seed)
    observed = _summary(per_gen)
    group_date = dates.groupby(groups).first()

    def one(i):
        # release groups keep their members together and trade dates as units
        rng = np.random.default_rng([seed, i])
        new = pd.Series(rng.permutation(group_date.to_numpy()), index=group_date.index)
        shuffled = pd.Series(new[groups[dates.index]].to_numpy(), index=dates.index)
        return _summary(logo_time_attribution(P[sel], gens, shuffled, groups, n_bins, seed))

    null = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(perms)))
    summary = {"n_generators": len(dates), "n_groups": int(groups.nunique()),
               "n_images": int(sel.sum()), "n_bins": n_bins,
               "chance_acc": 1 / n_bins, "perms": perms, **observed, "not_testable": None}
    for stat in ("acc", "llr"):
        summary[f"null_{stat}_mean"] = float(null[stat].mean())
        summary[f"null_{stat}_q95"] = float(null[stat].quantile(0.95))
        summary[f"p_{stat}"] = float((1 + (null[stat] >= observed[stat]).sum()) / (1 + perms))
    summary["null_mae_mean"] = float(null["mae"].mean())
    summary["p_mae"] = float((1 + (null["mae"] <= observed["mae"]).sum()) / (1 + perms))
    return summary, per_gen, null


def planted_power(P, index, family, deltas=(0.5, 1, 2, 4), reps=5, perms=99, n_bins=3, seed=0,
                  n_jobs=-1):
    """How large a drift T1 would detect *on this family's real features*.

    For each rep, a random unit direction u is drawn. Each image of a generator released at
    relative time t in [0, 1] is shifted by delta * s_u * t * u, with s_u the pooled
    within-generator SD along u. delta is therefore the total drift across the family's date
    range, in within-generator SDs. Whatever real structure the features already have (generator
    differences, origin effects, any true drift) stays in place. Returns one row per (delta, rep).
    """
    sel = _subset(index, family)
    sub = index[sel]
    gens = sub["generator"].to_numpy()
    raw = sub.drop_duplicates("generator").set_index("generator")["release_date"]
    dates = group_dates(raw, release_groups(raw))
    t = ((dates - dates.min()) / (dates.max() - dates.min())).astype(float)
    t_row = t[gens].to_numpy()
    base = P[sel]
    centred = base - pd.DataFrame(base).groupby(gens).transform("mean").to_numpy()
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng([seed, 7, rep])
        u = rng.normal(size=base.shape[1])
        u /= np.linalg.norm(u)
        s_u = float((centred @ u).std())
        for delta in deltas:
            planted = base + np.outer(delta * s_u * t_row, u)
            Pfull = np.zeros((len(index), base.shape[1]))
            Pfull[sel] = planted
            s, _, _ = t1_test(Pfull, index, family, n_bins, perms, seed + rep, None, n_jobs)
            rows.append(dict(family=family, delta=delta, rep=rep, llr=s["llr"],
                             null_llr_mean=s["null_llr_mean"], p_llr=s["p_llr"], acc=s["acc"]))
    return pd.DataFrame(rows)


def size_check(feature_sets, index, perms=1000, seed=0, n_jobs=-1, band_perms=None):
    """Is the T1 signal just native image size? Generator resolution rose over time in diffusion.

    For each family: (1) T1 on log native short side alone, a 1-D feature (with a 0.02 jitter so
    the per-bin variance is defined); (2) T1 on each feature set after regressing log size and
    its square out of every component, within the family. (1) says whether size *could* explain
    T1; (2) whether anything is left after a mean-level size adjustment. Neither removes size
    effects on the *spread* of the features, which the per-bin GMMs can also use.
    (3) For diffusion, (1) again inside each native-size band (SIZE_BANDS) at 2 and 3 bins: how
    much size still predicts the bin where the band specs test Phi (`band_perms` permutations).
    """
    log_size = np.log(native_short_side(index))
    rng = np.random.default_rng([seed, 11])
    jittered = np.zeros((len(index), 1))
    jittered[:, 0] = log_size + rng.normal(0, 0.02, len(index))
    rows = []
    for band in SIZE_BANDS:
        for nb in (2, 3):
            s, _, _ = t1_test(jittered, index, "diffusion", nb, band_perms or perms, seed, band,
                              n_jobs)
            rows.append(dict(family="diffusion", features="log native size only", subset=band,
                             size_r2=np.nan, **s))
    for fam in TIME_FAMILIES:
        sel = (index["generator_family"] == fam).to_numpy()
        s, _, _ = t1_test(jittered, index, fam, PRIMARY["n_bins"], perms, seed, None, n_jobs)
        rows.append(dict(family=fam, features="log native size only", subset="all",
                         size_r2=np.nan, **s))
        D = np.column_stack([np.ones(sel.sum()), log_size[sel], log_size[sel] ** 2])
        for name, P in feature_sets.items():
            beta = np.linalg.lstsq(D, P[sel], rcond=None)[0]
            Pr = P.copy()
            Pr[sel] = P[sel] - D @ beta
            r2 = 1 - (((Pr[sel] - Pr[sel].mean(0)) ** 2).sum()
                      / ((P[sel] - P[sel].mean(0)) ** 2).sum())
            s, _, _ = t1_test(Pr, index, fam, PRIMARY["n_bins"], perms, seed, None, n_jobs)
            rows.append(dict(family=fam, features=f"{name}, size regressed out", subset="all",
                             size_r2=r2, **s))
    return pd.DataFrame(rows)


# --- T2: divergence vs date gap ---------------------------------------------------------------

def cell_divergence(P, cells, seed=0):
    """Symmetric held-out divergence between every pair of cells (rows grouped by `cells`).

    Each cell's rows are split in half (seeded). A GMM fitted on one half scores the other half,
    and both directions are averaged, so no cell is ever scored on the rows it was fitted on:
        D(a, b) = 1/2 [ (l_a(a_test) - l_b(a_test)) + (l_b(b_test) - l_a(b_test)) ]
    with l_c the mean held-out log-likelihood under cell c's model. D ~ 0 for indistinguishable
    cells and can come out slightly negative from noise.
    """
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(cells))
    halves = {}
    for c in names:
        rows = rng.permutation(np.flatnonzero(cells == c))
        halves[c] = (rows[: len(rows) // 2], rows[len(rows) // 2:])
    models = {(c, h): fit_gmm_bic(P[halves[c][h]], seed) for c in names for h in (0, 1)}

    def ll(model_cell, test_cell):
        # the model trained on half h scores the *other* half of test_cell; for model_cell ==
        # test_cell that is a genuine held-out score, for a different cell either half is unseen
        return np.mean([models[(model_cell, h)].score_samples(P[halves[test_cell][1 - h]]).mean()
                        for h in (0, 1)])

    L = np.array([[ll(m, t) for t in names] for m in names])     # L[m, t]
    D = 0.5 * ((np.diag(L)[None, :] - L) + (np.diag(L)[None, :] - L).T)
    return names, D


def t2_test(P, index, family, perms=1000, seed=0):
    """Spearman(divergence, |delta release date|) over (generator x origin) cell pairs.

    Two versions share one permutation null (release dates shuffled among generators; a
    generator's cells move together): all pairs of cells from different generators, and only
    pairs that share an origin dataset -- the comparison origin cannot explain.
    """
    sel = (index["generator_family"] == family).to_numpy(copy=True)
    cell = (index["generator"] + " @ " + index["origin_dataset"]).to_numpy()
    counts = pd.Series(cell[sel]).value_counts()
    sel &= np.isin(cell, counts[counts >= MIN_CELL].index)
    if np.unique(cell[sel]).size < 2:
        # nothing to pair (e.g. a smoke-sized run): report the test as not testable
        empty = pd.DataFrame([{"pairs": name, "n_pairs": 0, "n_generators": 0, "spearman": np.nan,
                               "null_mean": np.nan, "null_q95": np.nan, "p": np.nan}
                              for name in ("all", "same origin")])
        return empty, pd.DataFrame(columns=["cell_a", "cell_b", "generator_a", "generator_b",
                                            "origin_a", "origin_b", "divergence", "gap_days",
                                            "same_generator", "same_origin"])
    names, D = cell_divergence(P[sel], cell[sel], seed)
    meta = index[sel].assign(cell=cell[sel]).drop_duplicates("cell").set_index("cell")
    gen = meta.loc[names, "generator"].to_numpy()
    org = meta.loc[names, "origin_dataset"].to_numpy()
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    gen_dates = group_dates(raw, groups)
    group_date = gen_dates.groupby(groups).first()

    iu = np.triu_indices(len(names), 1)
    # pairs inside one release group always have gap ~ 0; they say nothing about time
    diff_gen = groups[gen[iu[0]]].to_numpy() != groups[gen[iu[1]]].to_numpy()
    same_org = org[iu[0]] == org[iu[1]]
    d = D[iu]

    def stats(date_of):
        t = np.array([date_of[g].value for g in gen]) / 86400e9    # days
        gap = np.abs(t[iu[0]] - t[iu[1]])
        out = {}
        for name, mask in (("all", diff_gen), ("same origin", diff_gen & same_org)):
            out[name] = (spearmanr(d[mask], gap[mask]).statistic if mask.sum() >= 4 else np.nan)
        return out, gap

    observed, gap = stats(gen_dates)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(perms):
        new = pd.Series(rng.permutation(group_date.to_numpy()), index=group_date.index)
        nulls.append(stats(pd.Series(new[groups[gen_dates.index]].to_numpy(),
                                     index=gen_dates.index))[0])
    summary = []
    for name in observed:
        null = np.array([n[name] for n in nulls])
        mask = diff_gen if name == "all" else diff_gen & same_org
        testable = np.isfinite(observed[name])          # needs >= 4 pairs
        summary.append({"pairs": name, "n_pairs": int(mask.sum()),
                        "n_generators": int(len(set(gen[iu[0]][mask]) | set(gen[iu[1]][mask]))),
                        "spearman": observed[name],
                        "null_mean": float(np.nanmean(null)) if testable else np.nan,
                        "null_q95": float(np.nanquantile(null, 0.95)) if testable else np.nan,
                        "p": float((1 + np.sum(null >= observed[name])) / (1 + perms))
                        if testable else np.nan})
    pairs = pd.DataFrame({"cell_a": np.array(names)[iu[0]], "cell_b": np.array(names)[iu[1]],
                          "generator_a": gen[iu[0]], "generator_b": gen[iu[1]],
                          "origin_a": org[iu[0]], "origin_b": org[iu[1]],
                          "divergence": d, "gap_days": gap, "same_generator": ~diff_gen,
                          "same_origin": same_org})
    return pd.DataFrame(summary), pairs


# --- T3 / T4: what the components are ---------------------------------------------------------

def bin_components(P, index, family, n_bins=3, seed=0):
    """Fit each time bin's GMM on all its generators; one row per image with its component."""
    sel = _subset(index, family)
    sub = index[sel]
    raw = sub.drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    bins = time_bins(group_dates(raw, groups), n_bins, groups)
    row_bin = sub["generator"].map(bins).to_numpy()
    out = []
    for b in range(n_bins):
        rows = row_bin == b
        gm = fit_gmm_bic(P[sel][rows], seed, max_components=8)
        comp = gm.predict(P[sel][rows])
        part = sub[rows][["image_id", "generator", "origin_dataset", "source_format",
                          "release_date"]].assign(bin=b, component=comp,
                                                  n_components=gm.n_components)
        out.append(part)
    df = pd.concat(out, ignore_index=True)
    df["family"] = family
    return df


def generator_components(P, index, families, seed=0):
    """Per generator: BIC-chosen component count, and how its split lines up with metadata."""
    rows = []
    size = A.size_bucket(index).to_numpy()
    for fam in families:
        for g, part in index[index.generator_family == fam].groupby("generator"):
            r = part.index.to_numpy()
            gm = fit_gmm_bic(P[r], seed, max_components=MAX_COMPONENTS)
            comp = gm.predict(P[r])
            rec = {"family": fam, "generator": g, "release_date": part.release_date.iloc[0],
                   "n_images": len(r), "max_testable": max(1, min(MAX_COMPONENTS,
                                                                  len(r) // MIN_PER_COMPONENT)),
                   "n_components": gm.n_components}
            for name, lab in (("origin_dataset", part.origin_dataset.to_numpy()),
                              ("source_format", part.source_format.to_numpy()),
                              ("size", size[r])):
                rec[f"levels_{name}"] = len(np.unique(lab))
                rec[f"ari_{name}"] = (float(adjusted_rand_score(lab, comp))
                                      if gm.n_components > 1 and len(np.unique(lab)) > 1
                                      else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


# --- L1 / L2: lineage --------------------------------------------------------------------------

def load_lineage(path=LINEAGE_CSV):
    """selfdesc/lineage.csv indexed by generator. Labels were fixed before any result (D-011)."""
    return pd.read_csv(path).set_index("generator")


def _lineage_classes(gens_dates, groups, labels):
    """{generator: class index} for generators whose class spans >= 2 release groups.

    "unknown" is not a class. Returns (classes, class names)."""
    known = {g: labels[g] for g in gens_dates.index if g in labels and labels[g] != "unknown"}
    spans = pd.Series({lab: groups[[g for g in known if known[g] == lab]].nunique()
                       for lab in set(known.values())})
    names = sorted(spans[spans >= 2].index)
    return {g: names.index(lab) for g, lab in known.items() if lab in names}, names


def lineage_test(P, index, family, column="stage", perms=1000, seed=0, subset=None, n_jobs=-1,
                 lineage=None):
    """L1: attribute each held-out release group's generators to a lineage class.

    T1's machinery (logo_attribution) with lineage classes in place of time bins, so the two
    statistics are on one footing. Generators in "unknown" or in a class spanning a single
    release group are left out (a class must survive one group being held out). Null: lineage
    labels permuted among the kept generators, redrawn until every class again spans >= 2
    release groups (the observed labelling satisfies that by construction, so conditioning on
    it keeps the null exact). Returns (summary dict, per-generator table, null table).
    """
    lineage = load_lineage() if lineage is None else lineage
    sel = _subset(index, family, subset)
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    classes, names = _lineage_classes(raw, groups, lineage[column].to_dict())
    keep = sel & index["generator"].isin(list(classes)).to_numpy()
    base = {"column": column, "classes": names, "n_classes": len(names),
            "n_generators": len(classes), "n_groups": int(groups[list(classes)].nunique())
            if classes else 0, "n_images": int(keep.sum()), "perms": perms,
            "chance_acc": 1 / len(names) if names else np.nan}
    if len(names) < 2:
        return dict(base, acc=np.nan, llr=np.nan, p_llr=np.nan, p_acc=np.nan,
                    null_llr_mean=np.nan, null_llr_q95=np.nan, null_acc_mean=np.nan,
                    not_testable=f"{len(names)} lineage class(es) span >= 2 release groups"), \
            pd.DataFrame(), pd.DataFrame()
    gens = index.loc[keep, "generator"].to_numpy()
    Pk = P[keep]
    per_gen = logo_attribution(Pk, gens, classes, groups, len(names), raw, seed)
    per_gen["lineage"] = per_gen.true_bin.map(dict(enumerate(names)))
    observed = _summary(per_gen)
    order = list(classes)
    labels = np.array([classes[g] for g in order])

    def valid(perm_labels):
        lab = dict(zip(order, perm_labels))
        return all(groups[[g for g in order if lab[g] == b]].nunique() >= 2
                   for b in range(len(names)))

    def one(i):
        rng = np.random.default_rng([seed, 23, i])
        for _ in range(1000):
            shuffled = rng.permutation(labels)
            if valid(shuffled):
                break
        else:
            raise RuntimeError("no valid lineage permutation in 1000 draws")
        return _summary(logo_attribution(Pk, gens, dict(zip(order, shuffled)), groups,
                                         len(names), raw, seed))

    null = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(perms)))
    summary = dict(base, **{k: observed[k] for k in ("acc", "llr")}, not_testable=None)
    for stat in ("acc", "llr"):
        summary[f"null_{stat}_mean"] = float(null[stat].mean())
        summary[f"null_{stat}_q95"] = float(null[stat].quantile(0.95))
        summary[f"p_{stat}"] = float((1 + (null[stat] >= observed[stat]).sum()) / (1 + perms))
    return summary, per_gen, null


def _rank(x):
    return pd.Series(x).rank().to_numpy() / len(x)


def _ols(y, *cols):
    X = np.column_stack([np.ones(len(y)), *cols])
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def lineage_vs_date(pairs, index, family, column="stage", perms=1000, seed=0, lineage=None):
    """L2: does divergence between two generators follow lineage or date, each net of the other?

    On T2's held-out (generator x origin) cell divergences, pairs from different release groups
    with both lineages known. Rank regression of divergence on [different lineage, |delta date|]:
        b_lineage  > 0 if different-lineage pairs diverge more at equal date gap
        b_date     > 0 if pairs further apart in time diverge more at equal lineage
    (ranks scaled to 0..1, so a coefficient is a fraction of the rank range). Each has its own
    null: lineage labels permuted among generators for b_lineage, release dates permuted among
    release groups for b_date; the other variable stays as observed. Reported for all pairs and
    same-origin pairs. Also: mean divergence within / between lineages, and Spearman with date
    gap inside same-lineage pairs only.
    """
    lineage = load_lineage() if lineage is None else lineage
    labels = lineage[column].to_dict()
    sel = (index["generator_family"] == family).to_numpy()
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    gdates = group_dates(raw, groups)
    group_date = gdates.groupby(groups).first()
    known = pairs[(~pairs.same_generator)
                  & pairs.generator_a.map(labels).fillna("unknown").ne("unknown")
                  & pairs.generator_b.map(labels).fillna("unknown").ne("unknown")]
    rows = []
    for name, part in (("all", known), ("same origin", known[known.same_origin])):
        if len(part) < 6 or part.generator_a.map(labels).eq(part.generator_b.map(labels)).all():
            rows.append({"pairs": name, "n_pairs": len(part), "not_testable": "too few pairs"})
            continue
        y = _rank(part.divergence.to_numpy())
        ga, gb = part.generator_a.to_numpy(), part.generator_b.to_numpy()
        gens = sorted(set(ga) | set(gb))

        def diff_lineage(lab):
            return np.array([lab[a] != lab[b] for a, b in zip(ga, gb)], dtype=float)

        def gap_rank(dates):
            return _rank(np.abs((dates[ga].to_numpy() - dates[gb].to_numpy())
                                .astype("timedelta64[D]").astype(float)))

        x_lin, x_gap = diff_lineage(labels), gap_rank(gdates)
        b_lin, b_gap = _ols(y, x_lin, x_gap)
        rng = np.random.default_rng([seed, 31])
        null_lin, null_gap = [], []
        own = np.array([labels[g] for g in gens])
        for _ in range(perms):
            lab = dict(zip(gens, rng.permutation(own)))
            null_lin.append(_ols(y, diff_lineage(lab), x_gap)[0])
            new = pd.Series(rng.permutation(group_date.to_numpy()), index=group_date.index)
            dates = pd.Series(new[groups[gdates.index]].to_numpy(), index=gdates.index)
            null_gap.append(_ols(y, x_lin, gap_rank(dates))[1])
        null_lin, null_gap = np.array(null_lin), np.array(null_gap)
        same = x_lin == 0
        rho_same = (spearmanr(part.divergence[same], part.gap_days[same]).statistic
                    if same.sum() >= 4 else np.nan)
        rows.append({"pairs": name, "n_pairs": len(part), "n_generators": len(gens),
                     "n_same_lineage": int(same.sum()),
                     "mean_div_same_lineage": float(part.divergence[same].mean()),
                     "mean_div_diff_lineage": float(part.divergence[~same].mean()),
                     "b_lineage": float(b_lin), "null_b_lineage_q95": float(np.quantile(null_lin, .95)),
                     "p_lineage": float((1 + (null_lin >= b_lin).sum()) / (1 + perms)),
                     "b_date": float(b_gap), "null_b_date_q95": float(np.quantile(null_gap, .95)),
                     "p_date": float((1 + (null_gap >= b_gap).sum()) / (1 + perms)),
                     "spearman_gap_same_lineage": float(rho_same), "not_testable": None})
    return pd.DataFrame(rows).assign(family=family, column=column)


# --- T5: linear drift in generator means ---------------------------------------------------------

def _whitened_group_means(P, gens, groups, dates):
    """Release-group means of the generator means, in within-generator-SD (whitened) units.

    The pooled within-generator covariance W is estimated from rows centred on their own
    generator; Q = W^(-1/2) maps it to the identity, so a distance of 1 is one within-generator
    SD in any direction. Returns (group means (n_groups, g), group time t in [0, 1])."""
    frame = pd.DataFrame(P)
    centred = P - frame.groupby(gens).transform("mean").to_numpy()
    W = np.cov(centred, rowvar=False)
    vals, vecs = np.linalg.eigh(np.atleast_2d(W))
    Q = vecs @ np.diag(1 / np.sqrt(np.maximum(vals, 1e-12 * vals.max()))) @ vecs.T
    gen_means = frame.groupby(gens).mean() @ Q
    by_group = gen_means.groupby(groups[gen_means.index].to_numpy()).mean()
    gdate = dates.groupby(groups[dates.index]).first().loc[by_group.index]
    span = (gdate.max() - gdate.min()).days or 1
    t = ((gdate - gdate.min()).dt.days / span).to_numpy(dtype=float)
    return by_group.to_numpy(), t


def _drift(M, t):
    """||OLS slope of the rows of M on t||: the drift across t's 0..1 range (vectorised over t)."""
    t = np.atleast_2d(t)
    tc = t - t.mean(axis=1, keepdims=True)
    beta = (tc @ (M - M.mean(0))) / (tc ** 2).sum(axis=1, keepdims=True)
    return np.linalg.norm(beta, axis=1)


def drift_test(P, index, family, perms=1000, boot=1000, seed=0, subset=None):
    """T5: is there a linear trend in where a release group's generators sit, and how big?

    delta = || slope of the group means on release time ||, the drift over the family's whole
    date range in within-generator SDs (Mahalanobis). It is >= 0 by construction, so it is
    judged against its permutation null (dates permuted among release groups), and bounded by a
    group bootstrap (95th percentile; conservative, since the estimate is biased upward).
    tau = RMS scatter of group means around the line, per dimension, in the same units: the
    noise any time trend has to rise above, which only more release groups average down.
    A mean-only test: it cannot see a drift in spread, which T1's GMMs can.
    """
    sel = _subset(index, family, subset)
    gens = index.loc[sel, "generator"].to_numpy()
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    base = {"n_generators": len(raw), "n_groups": int(groups.nunique()),
            "n_images": int(sel.sum()), "perms": perms}
    if groups.nunique() < 4:
        return dict(base, not_testable=f"{groups.nunique()} release group(s) < 4"), None
    M, t = _whitened_group_means(P[sel], gens, groups, group_dates(raw, groups))
    delta = float(_drift(M, t)[0])
    rng = np.random.default_rng([seed, 41])
    null = _drift(M, np.array([rng.permutation(t) for _ in range(perms)]))
    idx = rng.integers(0, len(t), size=(boot, len(t)))
    ok = np.array([np.ptp(t[i]) > 0 for i in idx])
    boots = np.array([_drift(M[i], t[i])[0] for i in idx[ok]])
    tc = t - t.mean()
    beta = tc @ (M - M.mean(0)) / (tc ** 2).sum()
    resid = M - M.mean(0) - np.outer(tc, beta)
    n, g = M.shape
    tau = float(np.sqrt((resid ** 2).sum() / ((n - 2) * g)))
    summary = dict(base, g=g, delta=delta, null_delta_mean=float(null.mean()),
                   null_delta_q95=float(np.quantile(null, 0.95)),
                   p=float((1 + (null >= delta).sum()) / (1 + perms)),
                   boot_q95=float(np.quantile(boots, 0.95)), tau=tau, not_testable=None)
    # residuals rescaled by sqrt(n / (n - 2)) undo the shrinkage of a fitted line
    return summary, resid * np.sqrt(n / (n - 2))


def drift_power(resid, n_groups=(12, 20, 30, 50, 100), deltas=(0.5, 1, 2), reps=200, perms=199,
                seed=0):
    """How many release groups T5 needs: simulated families with the *observed* scatter.

    Each simulated family has N groups at evenly spaced times in [0, 1]; each group's mean is a
    residual vector resampled from the real family (so the scatter is the real one, not a
    Gaussian guess) plus delta * t * u along a random unit direction u. Power = share of reps
    with permutation p <= 0.05. One row per (N, delta).
    """
    rows = []
    g = resid.shape[1]
    for N in n_groups:
        t = np.linspace(0, 1, N)
        for delta in deltas:
            hits = 0
            for rep in range(reps):
                rng = np.random.default_rng([seed, 43, N, int(delta * 100), rep])
                u = rng.normal(size=g)
                u /= np.linalg.norm(u)
                M = resid[rng.integers(0, len(resid), N)] + np.outer(delta * t, u)
                obs = _drift(M, t)[0]
                null = _drift(M, np.array([rng.permutation(t) for _ in range(perms)]))
                hits += (1 + (null >= obs).sum()) / (1 + perms) <= 0.05
            rows.append({"n_groups": N, "delta": delta, "reps": reps, "power": hits / reps})
    return pd.DataFrame(rows)


# --- S1: separability, for the pipeline arms ------------------------------------------------------

def separability_test(Xb, index, g=10, perms=200, seed=0, subset=None):
    """How much family and generator information the features hold (R-001's two measures).

    family     5-NN balanced accuracy of family among fakes, on robust-scaled Phi -> PCA on the
               fakes -> g components (no centring: centring would remove what is measured).
    generator  5-NN balanced accuracy of generator *inside* (family x origin) cells, on the
               family-centred view (D-007), neighbours and shuffles kept inside cells.
    Both against permutation nulls. The pipeline arms compare these across input pipelines.
    """
    fake = (index["label"] == 1).to_numpy() & subset_mask(index, subset)
    sub = index[fake]
    Z = A.scale(Xb[fake])
    P, _ = A.centred_pca(Z - Z.mean(0), np.ones(len(Z), bool), n_components=min(g, Z.shape[1]))
    fam = A.separability(P[:, :g], sub["generator_family"].to_numpy(), n_perm=perms, seed=seed,
                         kmeans=False)
    Pc = features(Xb[fake], sub.reset_index(drop=True), "family", g)
    strata = (sub["generator_family"] + " | " + sub["origin_dataset"]).to_numpy()
    gen, _ = A.stratified_separability(Pc, sub["generator"].to_numpy(), strata, n_perm=perms,
                                       seed=seed)
    rows = [dict(measure="family", **{k: fam[k] for k in ("n", "levels", "knn_acc", "chance",
                                                          "null_mean", "null_q95", "p")})]
    if gen is not None:
        rows.append(dict(measure="generator within family x origin", levels=np.nan,
                         **{k: gen[k] for k in ("n", "knn_acc", "chance", "null_mean",
                                                "null_q95", "p")}, strata=gen["strata"]))
    return pd.DataFrame(rows).assign(g=g, subset=subset or "all")


# --- the run ----------------------------------------------------------------------------------

def _code_version():
    """Hash of the code every cached result depends on: change it and the cache is invalid."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("temporal.py", "analysis.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


class _Cache:
    """Per-block results on disk, so a killed multi-hour run resumes instead of restarting.

    Keys hash the block's parameters, the identity of the data it reads (manifest and FRE hashes
    of every config involved) and the analysis code itself, so a stale entry can never be served
    after either changes.
    """

    def __init__(self, directory, base):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base = dict(base, code=_code_version())
        self.hits = self.misses = 0

    def get(self, parts, compute):
        key = hashlib.sha256(json.dumps([self.base, parts], sort_keys=True, default=str)
                             .encode()).hexdigest()[:20]
        path = self.directory / f"{key}.pkl"
        if path.exists():
            self.hits += 1
            return pd.read_pickle(path)
        self.misses += 1
        value = compute()
        tmp = path.with_name(f".{key}.{os.getpid()}.tmp")
        pd.to_pickle(value, tmp)
        os.replace(tmp, path)
        return value


PRIMARY = dict(centring="family x origin", n_bins=3, g=10)
FEATURE_LABEL = {"full": "full Φ", "mean": "φ̄"}
DEFAULT_CONFIG = "K8_M11_B11_L3"
SUITES = ("full", "arms", "quick")

# Sensitivity specs beyond D-008's grid (D-011). Single origins: no centring choice can matter
# inside one origin dataset, so a signal that survives family-only centring but not these was
# origin. Two bins where one origin has only 4-5 release groups.
ORIGIN_SPECS = (("gan", "origin=Artifact", 3), ("gan", "origin=Forensynths", 3),
                ("diffusion", "origin=GenImage", 3), ("diffusion", "origin=Artifact", 2))


def _spec_id(**k):
    return " | ".join(f"{key}={k[key]}" for key in ("config", "family", "features", "centring",
                                                    "n_bins", "g", "subset"))


def output_dir(exp, config):
    """exp/temporal for the default config, exp/temporal_<config> for any other."""
    return Path(exp) / ("temporal" if config == DEFAULT_CONFIG else f"temporal_{config}")


def _t1_specs(suite, config, perms, grid_perms, config_perms, blocks, other_configs):
    """(kind, perms, config, family, features, centring, n_bins, g, subset) for one suite."""
    P = PRIMARY
    specs = [("primary", perms, config, fam, feat, P["centring"], P["n_bins"], P["g"], "all")
             for fam in TIME_FAMILIES for feat in ("full", "mean")]
    if suite == "arms":
        # ArtiFact ships every image already resized to 200 px and JPEG-compressed, so no input
        # pipeline of ours can restore what it removed: the arms are also read without it.
        specs += [("no ArtiFact", grid_perms, config, fam, feat, P["centring"], P["n_bins"],
                   P["g"], "origin!=Artifact") for fam in TIME_FAMILIES for feat in ("full", "mean")]
    if suite != "full":
        return specs
    for fam in TIME_FAMILIES:
        for feat in ("full", "mean"):
            for nb in (2, 4):
                specs.append(("bins", grid_perms, config, fam, feat, P["centring"], nb, P["g"],
                              "all"))
            for g in (5, 20):
                specs.append(("granularity", grid_perms, config, fam, feat, P["centring"],
                              P["n_bins"], g, "all"))
            specs.append(("centring", grid_perms, config, fam, feat, "family", P["n_bins"],
                          P["g"], "all"))
        for feat in blocks:
            if feat not in ("full", "mean"):
                specs.append(("block", grid_perms, config, fam, feat, P["centring"],
                              P["n_bins"], P["g"], "all"))
    for fam, subset, nb in ORIGIN_SPECS:
        for feat in ("full", "mean"):
            specs.append(("single origin", grid_perms, config, fam, feat, "family", nb, P["g"],
                          subset))
    for band in SIZE_BANDS:
        for nb in (2, 3):
            for feat in ("full", "mean"):
                specs.append(("size band", grid_perms, config, "diffusion", feat, P["centring"],
                              nb, P["g"], band))
    for cfg in other_configs:
        for fam in TIME_FAMILIES:
            for feat in ("full", "mean"):
                specs.append(("config", config_perms, cfg, fam, feat, P["centring"],
                              P["n_bins"], P["g"], "all"))
    return specs


def run(exp, config=DEFAULT_CONFIG, perms=1000, grid_perms=200, config_perms=100, seed=0,
        n_jobs=-1, suite="full", only="all", power_perms=99, sims=200, boot=1000):
    """Every block of `suite` for one config of one experiment directory.

    full   the 36k run: D-008's primary + grid, T2, power, size, T3/T4, plus D-011's origin and
           size-band specs, L1, L2, T5 (+ groups-needed simulation).
    arms   the pipeline arms (fakes-only manifests): primary + no-ArtiFact T1, L1 on the stage
           label, T5, S1 separability, T2.
    quick  primary + T2 + T3/T4 only (local check).
    """
    assert suite in SUITES, suite
    exp = Path(exp)
    out_dir = output_dir(exp, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    full, arms = suite == "full", suite == "arms"

    index, X, _, meta = A.load(exp, config)
    blocks = phi_blocks(X, meta["K"], meta["B"], _planes(meta))
    lineage = load_lineage()

    def data_id(cfg):
        m = json.loads((exp / "descriptions" / cfg / "meta.json").read_text())
        return [cfg, m["manifest_sha256"], m["fre_sha256"], m["rows"]]

    cache = _Cache(out_dir / "cache", {"seed": seed, "min_images": MIN_IMAGES,
                                       "min_cell": MIN_CELL, "window": RELEASE_WINDOW_DAYS,
                                       "lineage": hashlib.sha256(LINEAGE_CSV.read_bytes())
                                       .hexdigest()[:16]})
    feature_cache = {}

    def feats_for(cfg, feat, centring=PRIMARY["centring"], g=PRIMARY["g"]):
        key = (cfg, feat, centring, g)
        if key not in feature_cache:
            if cfg == config:
                Xb = blocks[feat]
            else:
                idx2, X2, _, m2 = A.load(exp, cfg)
                assert (idx2.image_id == index.image_id).all()
                Xb = phi_blocks(X2, m2["K"], m2["B"], _planes(m2))[feat]
            feature_cache[key] = features(Xb, index, centring, g)
        return feature_cache[key]

    def write(frames, name):
        frames = [f for f in frames if f is not None and len(f)]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(out_dir / name, index=False)

    def size_block():
        sets = {FEATURE_LABEL[f]: feats_for(config, f) for f in ("full", "mean")}
        cache.get(["size", data_id(config), perms, grid_perms],
                  lambda: size_check(sets, index, perms, seed, n_jobs, grid_perms)) \
            .to_parquet(out_dir / "t1_size.parquet", index=False)

    if only == "size":
        size_block()
        print(f"wrote {out_dir / 't1_size.parquet'} in {time.time() - started:.0f}s")
        return out_dir

    # --- T1 ---
    luma_family = [c for c in A.configs(exp) if c != config
                   and json.loads((exp / "descriptions" / c / "meta.json").read_text())
                   .get("channels", "luma") == meta.get("channels", "luma")]
    specs = _t1_specs(suite, config, perms, grid_perms, config_perms, blocks, luma_family)
    summaries, per_gens, nulls = [], [], []
    for n, (kind, p, cfg, fam, feat, centring, nb, g, subset) in enumerate(specs, 1):
        sid = _spec_id(config=cfg, family=fam, features=feat, centring=centring, n_bins=nb, g=g,
                       subset=subset)
        t = time.time()
        s, pg, nl = cache.get(
            ["t1", sid, p, data_id(cfg)],
            lambda: t1_test(feats_for(cfg, feat, centring, g), index, fam, nb, p, seed, subset,
                            n_jobs))
        summaries.append(dict(dict(spec=sid, kind=kind, config=cfg, family=fam, features=feat,
                                   centring=centring, g=g, subset=subset, origin=subset), **s))
        per_gens.append(pg.assign(spec=sid))
        nulls.append(nl.assign(spec=sid, perm=np.arange(len(nl))))
        tag = (f"not testable: {s['not_testable']}" if s.get("not_testable") else
               f"llr {s['llr']:+.3f} (null {s['null_llr_mean']:+.3f}, p {s['p_llr']:.3f}) "
               f"acc {s['acc']:.2f}")
        print(f"[{n}/{len(specs)}] {sid}: {tag} ({time.time() - t:.0f}s)", flush=True)

    t1 = pd.DataFrame(summaries)
    prim = t1.kind == "primary"
    t1["p_llr_holm"] = np.nan
    t1.loc[prim, "p_llr_holm"] = _holm(t1.loc[prim, "p_llr"].to_numpy())
    t1.to_parquet(out_dir / "t1_summary.parquet", index=False)
    write(per_gens, "t1_generators.parquet")
    write(nulls, "t1_null.parquet")

    # --- T2 (+ L2 on its pairs) ---
    t2s, t2p, l2 = [], [], []
    for fam in TIME_FAMILIES + (("inpainting", "other") if full else ()):
        for feat in ("full", "mean"):
            s, pairs = cache.get(["t2", fam, feat, perms, data_id(config)],
                                 lambda: t2_test(feats_for(config, feat), index, fam, perms,
                                                 seed))
            t2s.append(s.assign(family=fam, features=feat))
            t2p.append(pairs.assign(family=fam, features=feat))
            print(f"T2 {fam} {feat}:", s[["pairs", "n_pairs", "spearman", "p"]].round(3)
                  .to_dict("records"), flush=True)
            if full and fam in TIME_FAMILIES:
                for column in LINEAGE_COLUMNS:
                    res = cache.get(["l2", fam, feat, column, perms, data_id(config)],
                                    lambda: lineage_vs_date(pairs, index, fam, column, perms,
                                                            seed, lineage)
                                    if len(pairs) else
                                    pd.DataFrame([{"pairs": "all", "n_pairs": 0,
                                                   "not_testable": "no T2 pairs",
                                                   "family": fam, "column": column}]))
                    l2.append(res.assign(features=feat))
                    print(f"L2 {fam} {feat} {column}:",
                          res.drop(columns=["family", "column"], errors="ignore")
                          .filter(["pairs", "n_pairs", "b_lineage", "p_lineage", "b_date",
                                   "p_date"]).round(3).to_dict("records"), flush=True)
    write(t2s, "t2_summary.parquet")
    write(t2p, "t2_pairs.parquet")
    write(l2, "l2_summary.parquet")

    # --- L1 ---
    if full or arms:
        l1s, l1g, l1n = [], [], []
        for column in (LINEAGE_COLUMNS if full else ("stage",)):
            p_l1 = perms if column == "stage" else grid_perms
            for fam in TIME_FAMILIES:
                for feat in ("full", "mean"):
                    s, pg, nl = cache.get(
                        ["l1", fam, feat, column, p_l1, data_id(config)],
                        lambda: lineage_test(feats_for(config, feat), index, fam, column, p_l1,
                                             seed, None, n_jobs, lineage))
                    key = dict(family=fam, features=feat, column=column)
                    l1s.append(dict(key, **{k: v for k, v in s.items() if k != "column"}))
                    l1g.append(pg.assign(**key))
                    l1n.append(nl.assign(**key))
                    print(f"L1 {fam} {feat} {column}: " + (
                        f"not testable: {s['not_testable']}" if s.get("not_testable") else
                        f"classes {s['classes']} llr {s['llr']:+.3f} (null "
                        f"{s['null_llr_mean']:+.3f}, p {s['p_llr']:.3f}) acc {s['acc']:.2f}"),
                        flush=True)
        pd.DataFrame(l1s).assign(classes=lambda d: d.classes.map(", ".join)) \
            .to_parquet(out_dir / "l1_summary.parquet", index=False)
        write(l1g, "l1_generators.parquet")
        write(l1n, "l1_null.parquet")

    # --- T5 ---
    # Under family x origin centring the date-permutation null is not exchangeable: origin
    # datasets are tied to dates (ArtiFact early, SFHQ-T2I late), so centring on them removes
    # date-aligned variation and the observed drift lands *below* its null (found on the local
    # sample, D-011). Family-only centring lets origin pass as time. Inside one origin dataset
    # neither problem exists, so T5 runs under both centrings and within every single origin.
    if full or arms:
        t5s, t5p = [], []
        runs = [(c, sub) for c in ("family x origin", "family")
                for sub in (["all", "origin!=Artifact"] if arms else ["all"])]
        for fam in TIME_FAMILIES:
            fam_runs = runs + ([("family", sub) for f, sub, _ in ORIGIN_SPECS if f == fam]
                               + [(c, band) for band in SIZE_BANDS
                                  for c in ("family x origin", "family")]
                               if full and fam == "diffusion" else [])
            if full and fam == "gan":
                fam_runs += [("family", sub) for f, sub, _ in ORIGIN_SPECS if f == fam]
            for feat in ("full", "mean"):
                for centring, subset in fam_runs:
                    s, resid = cache.get(
                        ["t5", fam, feat, centring, subset, perms, boot, data_id(config)],
                        lambda: drift_test(feats_for(config, feat, centring), index, fam, perms,
                                           boot, seed, subset))
                    t5s.append(dict(family=fam, features=feat, centring=centring, subset=subset,
                                    **s))
                    print(f"T5 {fam} {feat} {centring} {subset}: " + (
                        f"not testable: {s['not_testable']}" if s.get("not_testable") else
                        f"delta {s['delta']:.2f} SD (null {s['null_delta_mean']:.2f}, "
                        f"p {s['p']:.3f}, 95% bound {s['boot_q95']:.2f}) tau {s['tau']:.2f}"),
                        flush=True)
                    if full and subset == "all" and resid is not None:
                        pw = cache.get(["t5 power", fam, feat, centring, sims, data_id(config)],
                                       lambda: drift_power(resid, reps=sims, seed=seed))
                        t5p.append(pw.assign(family=fam, features=feat, centring=centring,
                                             observed_groups=s["n_groups"]))
                        print(f"T5 power {fam} {feat} {centring}:",
                              pw.pivot(index="n_groups", columns="delta", values="power")
                              .round(2).to_dict("index"), flush=True)
        pd.DataFrame(t5s).to_parquet(out_dir / "t5_summary.parquet", index=False)
        write(t5p, "t5_power.parquet")

    # --- S1 (arms) ---
    if arms:
        s1 = []
        for feat in ("full", "mean"):
            for subset in ("all", "origin!=Artifact"):
                res = cache.get(["s1", feat, subset, grid_perms, data_id(config)],
                                lambda: separability_test(blocks[feat], index, PRIMARY["g"],
                                                          grid_perms, seed, subset))
                s1.append(res.assign(features=feat))
                print(f"S1 {feat} {subset}:", res[["measure", "knn_acc", "chance", "null_q95",
                                                   "p"]].round(3).to_dict("records"), flush=True)
        write(s1, "s1_separability.parquet")

    # --- planted power, size ---
    if full:
        power = []
        for fam in TIME_FAMILIES:
            for feat in ("full", "mean"):
                pw = cache.get(
                    ["power", fam, feat, power_perms, data_id(config)],
                    lambda: planted_power(feats_for(config, feat), index, fam, perms=power_perms,
                                          seed=seed, n_jobs=n_jobs)
                ).assign(features=feat)
                power.append(pw)
                print(f"power {fam} {feat}:", pw.groupby("delta").p_llr
                      .apply(lambda p: float((p <= 0.05).mean())).to_dict(), flush=True)
        write(power, "t1_power.parquet")
        size_block()

    # --- T3 / T4 ---
    if not arms:
        comps, gcomps = [], []
        for feat in ("full", "mean"):
            P = feats_for(config, feat)
            for fam in TIME_FAMILIES:
                comps.append(bin_components(P, index, fam, PRIMARY["n_bins"], seed)
                             .assign(features=feat))
            gcomps.append(generator_components(P, index, TIME_FAMILIES + ("inpainting", "other"),
                                               seed).assign(features=feat))
        write(comps, "t3_components.parquet")
        write(gcomps, "t4_generators.parquet")

    (out_dir / "meta.json").write_text(json.dumps({
        "config": config, "suite": suite, "perms": perms, "grid_perms": grid_perms,
        "config_perms": config_perms, "power_perms": power_perms, "sims": sims, "boot": boot,
        "seed": seed, "primary": PRIMARY, "min_images": MIN_IMAGES, "min_cell": MIN_CELL,
        "release_window_days": RELEASE_WINDOW_DAYS,
        "min_per_component": MIN_PER_COMPONENT, "max_components": MAX_COMPONENTS,
        "reg_covar": REG_COVAR, "covariance": "diag", "n_specs": len(specs),
        "lineage_sha256": hashlib.sha256(LINEAGE_CSV.read_bytes()).hexdigest(),
        "git_commit": _git_commit(), "seconds": round(time.time() - started, 1),
        "cache": {"hits": cache.hits, "misses": cache.misses, "code": cache.base["code"]},
        "rows": int(len(index)),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": "DECISIONS.md D-008, D-009, D-011"}, indent=2) + "\n")
    print(f"wrote {out_dir} in {time.time() - started:.0f}s")
    return out_dir


def _holm(p):
    """Holm step-down adjusted p-values."""
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _git_commit():
    from .extract import git_commit
    return git_commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="writes to <exp>/temporal (default config) or <exp>/temporal_<config>")
    parser.add_argument("--suite", choices=SUITES, default="full",
                        help="full: the 36k run; arms: the pipeline arms; quick: primary only")
    parser.add_argument("--perms", type=int, default=1000, help="primary specs, T2, L1/L2, T5")
    parser.add_argument("--grid-perms", type=int, default=200, help="sensitivity grid")
    parser.add_argument("--config-perms", type=int, default=100, help="other configs")
    parser.add_argument("--power-perms", type=int, default=99, help="planted-drift power runs")
    parser.add_argument("--sims", type=int, default=200, help="T5 groups-needed simulations")
    parser.add_argument("--boot", type=int, default=1000, help="T5 group bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="same as --suite quick")
    parser.add_argument("--only", choices=["all", "size"], default="all",
                        help="'size': only (re)compute the native-size check, t1_size.parquet")
    args = parser.parse_args()
    run(args.exp, args.config, args.perms, args.grid_perms, args.config_perms, args.seed,
        args.jobs, "quick" if args.quick else args.suite, args.only, args.power_perms, args.sims,
        args.boot)


if __name__ == "__main__":
    main()
