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

Features are computed without any date information (scaling, centring, PCA are unsupervised), so
fitting them on all rows does not leak the quantity being tested.
"""

import argparse
import json
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

__all__ = ["phi_blocks", "features", "release_groups", "time_bins", "fit_gmm_bic",
           "logo_time_attribution",
           "t1_test", "planted_power", "size_check", "cell_divergence", "t2_test",
           "bin_components",
           "generator_components"]

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


# --- features --------------------------------------------------------------------------------

def phi_blocks(X, K, B):
    """Named column blocks of Phi.

    "mean" is phi-bar = mean over the K residual channels of phi_k. The channels are ~0.93
    correlated, so phi-bar is the well-conditioned shared filter and "deviations" (phi_k - phi-bar)
    hold the large cancelling coefficients the joint fit produces. phi-bar is split further by
    neighbourhood ring (Chebyshev distance of the offset from the centre pixel) and by offset
    parity (both coordinates even: the lattice a 2x upsampling stage acts on).
    """
    n = len(X)
    n_feat = B * B - 1
    phi = X.reshape(n, K, n_feat)
    mean = phi.mean(axis=1)
    c = B // 2
    offsets = [(i - c, j - c) for i in range(B) for j in range(B) if (i, j) != (c, c)]
    ring = np.array([max(abs(a), abs(b)) for a, b in offsets])
    even = np.array([a % 2 == 0 and b % 2 == 0 for a, b in offsets])
    blocks = {"full": X, "mean": mean, "deviations": (phi - mean[:, None]).reshape(n, -1),
              "mean ring 1": mean[:, ring == 1], "mean ring 2": mean[:, ring == 2],
              "mean ring 3+": mean[:, ring >= 3],
              "mean even offsets": mean[:, even], "mean odd offsets": mean[:, ~even]}
    return blocks


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


def logo_time_attribution(P, gens, dates, groups, n_bins, seed=0):
    """Leave-one-release-group-out time-bin attribution. One row per generator.

    `dates` must already be group dates (every member of a group shares one), so bins never split
    a group. Each group is held out whole; each of its generators is then scored on its own.
    Columns: generator, group, date, true_bin, ll_<b> (mean log-likelihood per image under bin b's
    GMM), pred_bin, abs_err, llr (true bin minus the mean of the other bins).
    """
    bins = time_bins(dates, n_bins, groups)
    groups_per_bin = pd.Series({b: groups[[g for g in bins if bins[g] == b]].nunique()
                                for b in range(n_bins)})
    assert groups_per_bin.min() >= 2, \
        f"every bin needs >= 2 release groups so one can be held out: {groups_per_bin.to_dict()}"
    row_bin = np.array([bins[g] for g in gens])
    row_group = groups[gens].to_numpy()
    rows = []
    for grp in np.unique(row_group):
        held = row_group == grp
        models = [fit_gmm_bic(P[~held & (row_bin == b)], seed) for b in range(n_bins)]
        for g in np.unique(gens[held]):
            mine = gens == g
            ll = np.array([m.score_samples(P[mine]).mean() for m in models])
            true = bins[g]
            pred = int(np.argmax(ll))
            rows.append(dict(generator=g, group=int(grp), date=dates[g], true_bin=true,
                             pred_bin=pred, abs_err=abs(pred - true),
                             llr=float(ll[true] - np.delete(ll, true).mean()),
                             **{f"ll_{b}": float(v) for b, v in enumerate(ll)}))
    return pd.DataFrame(rows)


def _summary(table):
    """Averaged over release groups (the independent units), each group = mean of its members."""
    by_group = table.assign(hit=table.pred_bin == table.true_bin).groupby("group")
    return {"acc": float(by_group.hit.mean().mean()), "mae": float(by_group.abs_err.mean().mean()),
            "llr": float(by_group.llr.mean().mean())}


def _subset(index, family, origin=None):
    """Row mask for a family (optionally one origin), keeping generators with >= MIN_IMAGES rows."""
    sel = (index["generator_family"] == family).to_numpy(copy=True)
    if origin is not None:
        sel &= (index["origin_dataset"] == origin).to_numpy()
    counts = index[sel].groupby("generator").size()
    keep = counts[counts >= MIN_IMAGES].index
    return sel & index["generator"].isin(keep).to_numpy()


def t1_test(P, index, family, n_bins=3, perms=1000, seed=0, origin=None, n_jobs=-1):
    """T1 with its generator-level permutation null. Returns (summary dict, per-generator, nulls)."""
    sel = _subset(index, family, origin)
    gens = index.loc[sel, "generator"].to_numpy()
    raw = index[sel].drop_duplicates("generator").set_index("generator")["release_date"]
    groups = release_groups(raw)
    dates = group_dates(raw, groups)
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
               "chance_acc": 1 / n_bins, "perms": perms, **observed}
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


def size_check(feature_sets, index, perms=1000, seed=0, n_jobs=-1):
    """Is the T1 signal just native image size? Generator resolution rose over time in diffusion.

    For each family: (1) T1 on log native short side alone, a 1-D feature (with a 0.02 jitter so
    the per-bin variance is defined); (2) T1 on each feature set after regressing log size and
    its square out of every component, within the family. (1) says whether size *could* explain
    T1; (2) whether anything is left after a mean-level size adjustment. Neither removes size
    effects on the *spread* of the features, which the per-bin GMMs can also use.
    """
    log_size = np.log(np.minimum(index["source_width"], index["source_height"])
                      .to_numpy(dtype=float))
    rows = []
    for fam in TIME_FAMILIES:
        sel = (index["generator_family"] == fam).to_numpy()
        rng = np.random.default_rng([seed, 11])
        P = np.zeros((len(index), 1))
        P[:, 0] = log_size + rng.normal(0, 0.02, len(index))
        s, _, _ = t1_test(P, index, fam, PRIMARY["n_bins"], perms, seed, None, n_jobs)
        rows.append(dict(family=fam, features="log native size only", size_r2=np.nan, **s))
        D = np.column_stack([np.ones(sel.sum()), log_size[sel], log_size[sel] ** 2])
        for name, P in feature_sets.items():
            beta = np.linalg.lstsq(D, P[sel], rcond=None)[0]
            Pr = P.copy()
            Pr[sel] = P[sel] - D @ beta
            r2 = 1 - (((Pr[sel] - Pr[sel].mean(0)) ** 2).sum()
                      / ((P[sel] - P[sel].mean(0)) ** 2).sum())
            s, _, _ = t1_test(Pr, index, fam, PRIMARY["n_bins"], perms, seed, None, n_jobs)
            rows.append(dict(family=fam, features=f"{name}, size regressed out", size_r2=r2, **s))
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


# --- the run ----------------------------------------------------------------------------------

PRIMARY = dict(centring="family x origin", n_bins=3, g=10)
FEATURE_LABEL = {"full": "full Φ", "mean": "φ̄"}


def _spec_id(**k):
    return " | ".join(f"{key}={k[key]}" for key in ("config", "family", "features", "centring",
                                                    "n_bins", "g", "origin"))


def run(exp, config="K8_M11_B11_L3", perms=1000, grid_perms=200, config_perms=100, seed=0,
        n_jobs=-1, quick=False, only="all"):
    exp = Path(exp)
    out_dir = exp / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    index, X, _, meta = A.load(exp, config)
    blocks = phi_blocks(X, meta["K"], meta["B"])

    if only == "size":
        sets = {FEATURE_LABEL[f]: features(blocks[f], index, PRIMARY["centring"], PRIMARY["g"])
                for f in ("full", "mean")}
        size_check(sets, index, perms, seed, n_jobs).to_parquet(out_dir / "t1_size.parquet",
                                                                 index=False)
        print(f"wrote {out_dir / 't1_size.parquet'} in {time.time() - started:.0f}s")
        return out_dir

    specs = []   # (kind, perms, config, family, features, centring, n_bins, g, origin)
    for fam in TIME_FAMILIES:
        for feat in ("full", "mean"):
            specs.append(("primary", perms, config, fam, feat, PRIMARY["centring"],
                          PRIMARY["n_bins"], PRIMARY["g"], None))
    if not quick:
        for fam in TIME_FAMILIES:
            for feat in ("full", "mean"):
                for nb in (2, 4):
                    specs.append(("bins", grid_perms, config, fam, feat, PRIMARY["centring"], nb,
                                  PRIMARY["g"], None))
                for g in (5, 20):
                    specs.append(("granularity", grid_perms, config, fam, feat,
                                  PRIMARY["centring"], PRIMARY["n_bins"], g, None))
                specs.append(("centring", grid_perms, config, fam, feat, "family",
                              PRIMARY["n_bins"], PRIMARY["g"], None))
            for feat in blocks:
                if feat not in ("full", "mean"):
                    specs.append(("block", grid_perms, config, fam, feat, PRIMARY["centring"],
                                  PRIMARY["n_bins"], PRIMARY["g"], None))
        for feat in ("full", "mean"):
            specs.append(("single origin", grid_perms, config, "gan", feat, "family",
                          PRIMARY["n_bins"], PRIMARY["g"], "Artifact"))
        for cfg in A.configs(exp):
            if cfg == config:
                continue
            for fam in TIME_FAMILIES:
                for feat in ("full", "mean"):
                    specs.append(("config", config_perms, cfg, fam, feat, PRIMARY["centring"],
                                  PRIMARY["n_bins"], PRIMARY["g"], None))

    cache = {}

    def feats_for(cfg, feat, centring, g):
        key = (cfg, feat, centring, g)
        if key not in cache:
            if cfg == config:
                Xb = blocks[feat]
            else:
                idx2, X2, _, m2 = A.load(exp, cfg)
                assert (idx2.image_id == index.image_id).all()
                Xb = phi_blocks(X2, m2["K"], m2["B"])[feat]
            cache[key] = features(Xb, index, centring, g)
        return cache[key]

    summaries, per_gens, nulls = [], [], []
    for n, (kind, p, cfg, fam, feat, centring, nb, g, origin) in enumerate(specs, 1):
        sid = _spec_id(config=cfg, family=fam, features=feat, centring=centring, n_bins=nb, g=g,
                       origin=origin or "all")
        t = time.time()
        P = feats_for(cfg, feat, centring, g)
        s, pg, nl = t1_test(P, index, fam, nb, p, seed, origin, n_jobs)
        summaries.append(dict(dict(spec=sid, kind=kind, config=cfg, family=fam, features=feat,
                                   centring=centring, g=g, origin=origin or "all"), **s))
        per_gens.append(pg.assign(spec=sid))
        nulls.append(nl.assign(spec=sid, perm=np.arange(len(nl))))
        print(f"[{n}/{len(specs)}] {sid}: llr {s['llr']:+.3f} (null {s['null_llr_mean']:+.3f}, "
              f"p {s['p_llr']:.3f}) acc {s['acc']:.2f} ({time.time() - t:.0f}s)", flush=True)

    t1 = pd.DataFrame(summaries)
    prim = t1.kind == "primary"
    t1["p_llr_holm"] = np.nan
    t1.loc[prim, "p_llr_holm"] = _holm(t1.loc[prim, "p_llr"].to_numpy())
    t1.to_parquet(out_dir / "t1_summary.parquet", index=False)
    pd.concat(per_gens, ignore_index=True).to_parquet(out_dir / "t1_generators.parquet", index=False)
    pd.concat(nulls, ignore_index=True).to_parquet(out_dir / "t1_null.parquet", index=False)

    t2s, t2p = [], []
    for fam in TIME_FAMILIES + (("inpainting", "other") if not quick else ()):
        for feat in ("full", "mean"):
            P = feats_for(config, feat, PRIMARY["centring"], PRIMARY["g"])
            s, pairs = t2_test(P, index, fam, perms, seed)
            t2s.append(s.assign(family=fam, features=feat))
            t2p.append(pairs.assign(family=fam, features=feat))
            print(f"T2 {fam} {feat}:", s[["pairs", "n_pairs", "spearman", "p"]].round(3)
                  .to_dict("records"), flush=True)
    pd.concat(t2s, ignore_index=True).to_parquet(out_dir / "t2_summary.parquet", index=False)
    pd.concat(t2p, ignore_index=True).to_parquet(out_dir / "t2_pairs.parquet", index=False)

    power = []
    if not quick:
        for fam in TIME_FAMILIES:
            for feat in ("full", "mean"):
                P = feats_for(config, feat, PRIMARY["centring"], PRIMARY["g"])
                pw = planted_power(P, index, fam, seed=seed, n_jobs=n_jobs).assign(features=feat)
                power.append(pw)
                print(f"power {fam} {feat}:", pw.groupby("delta").p_llr
                      .apply(lambda p: float((p <= 0.05).mean())).to_dict(), flush=True)
        pd.concat(power, ignore_index=True).to_parquet(out_dir / "t1_power.parquet", index=False)

    if not quick:
        sets = {FEATURE_LABEL[f]: feats_for(config, f, PRIMARY["centring"], PRIMARY["g"])
                for f in ("full", "mean")}
        size_check(sets, index, perms, seed, n_jobs).to_parquet(out_dir / "t1_size.parquet",
                                                                 index=False)

    comps, gcomps = [], []
    for feat in ("full", "mean"):
        P = feats_for(config, feat, PRIMARY["centring"], PRIMARY["g"])
        for fam in TIME_FAMILIES:
            comps.append(bin_components(P, index, fam, PRIMARY["n_bins"], seed)
                         .assign(features=feat))
        gcomps.append(generator_components(P, index, TIME_FAMILIES + ("inpainting", "other"),
                                           seed).assign(features=feat))
    pd.concat(comps, ignore_index=True).to_parquet(out_dir / "t3_components.parquet", index=False)
    pd.concat(gcomps, ignore_index=True).to_parquet(out_dir / "t4_generators.parquet", index=False)

    (out_dir / "meta.json").write_text(json.dumps({
        "config": config, "perms": perms, "grid_perms": grid_perms, "config_perms": config_perms,
        "seed": seed, "primary": PRIMARY, "min_images": MIN_IMAGES, "min_cell": MIN_CELL,
        "release_window_days": RELEASE_WINDOW_DAYS,
        "min_per_component": MIN_PER_COMPONENT, "max_components": MAX_COMPONENTS,
        "reg_covar": REG_COVAR, "covariance": "diag", "n_specs": len(specs), "quick": quick,
        "git_commit": _git_commit(), "seconds": round(time.time() - started, 1),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": "DECISIONS.md D-008"}, indent=2) + "\n")
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
    parser.add_argument("--config", default="K8_M11_B11_L3")
    parser.add_argument("--perms", type=int, default=1000, help="primary specs and T2")
    parser.add_argument("--grid-perms", type=int, default=200, help="sensitivity grid")
    parser.add_argument("--config-perms", type=int, default=100, help="other configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="primary specs only (smoke test)")
    parser.add_argument("--only", choices=["all", "size"], default="all",
                        help="'size': only (re)compute the native-size check, t1_size.parquet")
    args = parser.parse_args()
    run(args.exp, args.config, args.perms, args.grid_perms, args.config_perms, args.seed,
        args.jobs, args.quick, args.only)


if __name__ == "__main__":
    main()
