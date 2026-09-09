"""Per-family score models: ridge fusion of the panel, then PAV calibration, then ELUB.

Fusion is PER FAMILY, and that is a correctness requirement rather than a preference. PAV
output is monotone by construction, so if every family's model were phi_g(s) for the SAME
global scalar s, then max_g LR_g would itself be a monotone function of s: the G<=T aggregator
would add no information, T*(F) would be a relabelling of one number, and family attribution
would be impossible. Global fusion is kept only as a comparator.
"""

import numpy as np
import pandas as pd

from .bounds import count_bound, elub
from .core import fuse, pav_lr
from .data import Z, kfolds
from .metrics import cllr

__all__ = ["L2_GRID", "family_xy", "fit_family", "pick_l2", "fit_all", "lr_matrix",
           "fit_generator_models", "fit_truncated_models", "cross_fit"]

L2_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def family_xy(df, family):
    """Design matrices for LR_g. H_d is authentic ONLY.

    Other families' fakes are neither hypothesis for this model -- the question LR_g answers is
    "family g versus authentic" -- and folding them into the denominator collapses the bounds.
    """
    A = df.loc[df.label == 0, Z].values
    fk = df[(df.label == 1) & (df.generator_family == family)]
    assert (fk.generator_family == family).all(), "H_p must be exactly one family"
    return A, fk[Z].values


def _model_from(A, F, l2):
    fused, clf = fuse(A, F, l2)
    cal = pav_lr(fused(A), fused(F))
    model = lambda P, c=cal, f=fused: c(f(np.asarray(P, float)))
    y = np.r_[np.zeros(len(A)), np.ones(len(F))]
    lr = model(np.vstack([A, F]))
    lo, hi = elub(lr, y)
    return dict(model=model, fused=fused, cal=cal, clf=clf, lo=lo, hi=hi,
                n_tr=len(F), lr_tr=lr, y_tr=y, l2=l2, s_auth=fused(A), s_fake=fused(F))


def fit_family(df, family, l2):
    """Fit one family's fusion + calibrator + ELUB bounds on df."""
    return _model_from(*family_xy(df, family), l2)


def pick_l2(df, family, grid=L2_GRID, k=5, seed=0):
    """lambda by inner k-fold CV within df, scored by out-of-fold Cllr.

    LRs are clipped to the count bound rather than to a fitted ELUB: the count bound is
    data-independent given n, so the criterion stays finite without fitting ELUB inside every
    fold. Returns (best_lambda, {lambda: cv_cllr}).
    """
    auth = df[df.label == 0].reset_index(drop=True)
    fake = df[(df.label == 1) & (df.generator_family == family)].reset_index(drop=True)
    kk = int(min(k, len(fake)))
    fa, ff = kfolds(auth, kk, "source", seed), kfolds(fake, kk, "generator", seed)
    out = {}
    for l2 in grid:
        got = []
        for f in range(kk):
            inner = pd.concat([auth[fa != f], fake[ff != f]])
            te_a, te_f = auth[fa == f][Z].values, fake[ff == f][Z].values
            if len(te_a) == 0 or len(te_f) == 0:
                continue
            m = fit_family(inner, family, l2)
            lo, hi = count_bound(int((inner.label == 1).sum()), int((inner.label == 0).sum()))
            lr = np.clip(m["model"](np.vstack([te_a, te_f])), lo, hi)
            got.append(cllr(lr, np.r_[np.zeros(len(te_a)), np.ones(len(te_f))]))
        out[l2] = float(np.mean(got))
    return min(out, key=out.get), out


def fit_all(df, families, mode="per_family", l2s=None):
    """mode='per_family': one ridge fusion per family (primary).
       mode='global':     one fusion for all fakes, per-family PAV on that single scalar."""
    if mode == "global":
        A_all = df.loc[df.label == 0, Z].values
        F_all = df.loc[df.label == 1, Z].values
        fused, _ = fuse(A_all, F_all, 1.0)
        models = {}
        for g in families:
            _, F = family_xy(df, g)
            cal = pav_lr(fused(A_all), fused(F))
            model = lambda P, c=cal, f=fused: c(f(np.asarray(P, float)))
            y = np.r_[np.zeros(len(A_all)), np.ones(len(F))]
            lr = model(np.vstack([A_all, F]))
            lo, hi = elub(lr, y)
            models[g] = dict(model=model, fused=fused, cal=cal, lo=lo, hi=hi, n_tr=len(F),
                             lr_tr=lr, y_tr=y, l2=1.0, s_auth=fused(A_all), s_fake=fused(F))
        return models
    return {g: fit_family(df, g, l2s[g]) for g in families}


def lr_matrix(models, df, keys, bound=True):
    """n x |keys| matrix of (ELUB-bounded) LR_g for every row of df."""
    P = df[Z].values
    out = np.empty((len(df), len(keys)))
    for j, g in enumerate(keys):
        v = models[g]["model"](P)
        out[:, j] = np.clip(v, models[g]["lo"], models[g]["hi"]) if bound else v
    return out


def fit_generator_models(df, gens, l2=1.0):
    """One model per generator, for the granularity sensitivity.

    lambda is fixed rather than cross-validated: per-generator CV on 8 training images selects
    noise. Note the source-count bound is [1, 1] for EVERY generator model, so finer temporal
    resolution means fewer sources per model and therefore less supportable evidence.
    """
    models = {}
    A = df.loc[df.label == 0, Z].values
    for name in gens:
        F = df.loc[(df.label == 1) & (df.generator == name), Z].values
        models[name] = _model_from(A, F, l2)
    return models


def fit_truncated_models(df, generators, families, l2s):
    """Availability-truncated family models: {(family, n_generators_so_far): model}.

    LR_g^{<=T} is fitted only on the generators in family g released by T, which removes the
    temporal leak of fitting a family's model on generators released AFTER T. There are exactly
    as many distinct models as generators -- one per generator-addition event -- not one per
    (family, date) pair.
    """
    out = {}
    A = df.loc[df.label == 0, Z].values
    for g in families:
        members = generators[generators.fam == g].sort_values("date")
        for j in range(1, len(members) + 1):
            subset = list(members.index[:j])
            F = df.loc[(df.label == 1) & (df.generator.isin(subset)), Z].values
            out[(g, j)] = _model_from(A, F, l2s[g])
    return out


def cross_fit(M, families, l2s, k=5, seed=0):
    """Out-of-fold LR_g for every image, folds stratified by (label, generator).

    The fixed split leaves the one-generator families with 9 validation fakes, at which point
    Cllr is not reportable. This never calibrates on an image it scores. Returns (Mi, OOF).
    """
    Mi = M.reset_index(drop=True).copy()
    Mi["stratum"] = np.where(Mi.label == 0, "auth:" + Mi.source, "gen:" + Mi.generator)
    Mi["fold"] = kfolds(Mi, k, "stratum", seed)
    oof = np.full((len(Mi), len(families)), np.nan)
    for f in range(k):
        inner, held = Mi[Mi.fold != f], Mi[Mi.fold == f]
        oof[held.index.values, :] = lr_matrix(
            fit_all(inner, families, "per_family", l2s), held, families)
    assert not np.isnan(oof).any(), "every image must receive an out-of-fold LR"
    return Mi, oof
