"""Tables. Each function returns a DataFrame or dict the notebook displays.

Kept separate from metrics.py: that module holds the statistics, this one holds the particular
cross-tabulations this experiment reports.
"""

import numpy as np
import pandas as pd

from .aggregate import build_aggregators, crossing, localization_error
from .bounds import adopt_bounds, count_bound, elub, source_bound
from .core import fuse, logistic_lr, pav_lr
from .data import Z
from .metrics import (check_pav_identity, cllr, cllr_decomp, cllr_min,
                      rates_misleading)
from .models import family_xy, fit_all, lr_matrix

__all__ = ["geometry_floor", "identity_table", "family_table", "cost_table", "rme_table",
           "leakage_matrix", "blind_diagnosis", "blind_battery", "multiplicity", "tstar_table",
           "calibrator_comparison", "crossfit_table", "residualization_slopes",
           "summary_table"]


def _bounded(model, X):
    return np.clip(model["model"](X), model["lo"], model["hi"])


def _family_val(va, family):
    sub = va[(va.label == 0) | (va.generator_family == family)]
    return sub, (sub.generator_family == family).astype(int).values


def geometry_floor(tr, va, families):
    """The same PAV/ELUB/Cllr machinery driven by one metadata field: the crop-area fraction.

    Zero pixels. Everything the panel reports should be read against this.
    """
    rows = []
    for g in families:
        a_tr = tr.loc[tr.label == 0, "logfrac"].values
        f_tr = tr.loc[(tr.label == 1) & (tr.generator_family == g), "logfrac"].values
        cal = pav_lr(a_tr, f_tr)
        lo, hi = elub(cal(np.r_[a_tr, f_tr]),
                      np.r_[np.zeros(len(a_tr)), np.ones(len(f_tr))])
        sub, y = _family_val(va, g)
        c, cmin, ccal = cllr_decomp(np.clip(cal(sub.logfrac.values), lo, hi), y)
        rows.append({"family": g, "n_val_fake": int(y.sum()), "elub": f"[{lo:.3f}, {hi:.2f}]",
                     "Cllr": c, "Cllr_min": cmin, "Cllr_cal": ccal})
    return pd.DataFrame(rows).set_index("family")


def identity_table(mods, families):
    return pd.DataFrame([check_pav_identity(mods[g]["s_auth"], mods[g]["s_fake"], g)
                         for g in families]).set_index("family")


def family_table(mods, tr, va, families, fam_ngen, floor=None):
    """Per-family bounds and Cllr decomposition, with all three bounds and the adopted one."""
    n_auth_tr = int((tr.label == 0).sum())
    rows = []
    for g in families:
        m = mods[g]
        sub, y = _family_val(va, g)
        c, cmin, ccal = cllr_decomp(_bounded(m, sub[Z].values), y)
        cb, sb = count_bound(m["n_tr"], n_auth_tr), source_bound(fam_ngen[g])
        ad = adopt_bounds((m["lo"], m["hi"]), cb, sb)
        row = {"family": g, "ngen": fam_ngen[g], "n_tr": m["n_tr"], "n_val": int(y.sum()),
               "lambda": m["l2"], "ELUB": f"[{m['lo']:.3f}, {m['hi']:.1f}]",
               "span_dex": np.log10(m["hi"] / m["lo"]),
               "count_bd": f"[{cb[0]:.3f}, {cb[1]:.0f}]",
               "source_bd": f"[{sb[0]:.3f}, {sb[1]:.0f}]",
               "adopted": f"[{ad[0]:.3f}, {ad[1]:.1f}]",
               "Cllr": c, "Cllr_min": cmin, "Cllr_cal": ccal}
        if floor is not None:
            row["Cllr_floor(geom)"] = floor.loc[g, "Cllr"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("family")


def cost_table(mods, families, thetas, fpr_grid):
    """What each family charges on authentic images to report a given LR.

    Prices every threshold in the currency that matters, so nothing rests on reading a log
    axis by eye.
    """
    rows = []
    for g in families:
        m = mods[g]
        thr = np.percentile(m["s_auth"], 100.0 - fpr_grid)
        lr = np.clip(m["cal"](thr), m["lo"], m["hi"])
        row = {"family": g, "max LR reachable": lr.max()}
        for th in thetas:
            ok = np.flatnonzero(lr >= th)
            row[f"authentic FPR to reach LR>={th:g}"] = (
                f"{fpr_grid[ok[-1]]:.2f}%" if len(ok) else "unreachable")
        rows.append(row)
    return pd.DataFrame(rows).set_index("family")


def rme_table(mods, va, families, thresholds=(1.0, 4.0, 10.0)):
    rows = []
    for g in families:
        m = mods[g]
        sub, y = _family_val(va, g)
        lr = _bounded(m, sub[Z].values)
        rows.append({"family": g, "n_val_fake": int(y.sum()),
                     "distinct_LR": len(np.unique(lr)),
                     **rates_misleading(lr, y, thresholds)})
    return pd.DataFrame(rows).set_index("family")


def leakage_matrix(L, va, families):
    """Geometric-mean LR_g by TRUE family. A usable set of models has a dominant diagonal."""
    lab = va.generator_family.values
    rows = [pd.Series(np.exp(np.log(np.maximum(L[lab == t], 1e-12)).mean(0)),
                      index=families, name=f"{t} (n={(lab == t).sum()})")
            for t in ["authentic"] + list(families)]
    out = pd.DataFrame(rows)
    out.columns.name = "model ->"
    return out


def blind_diagnosis(leak, va, families):
    """Which families' own model wins on their own images, and whether it wins above 1."""
    lab = va.generator_family.values
    rows = []
    for g in families:
        row = leak.loc[f"{g} (n={(lab == g).sum()})"]
        rows.append({"family": g, "own model": row[g], "argmax": row.idxmax(),
                     "diagonal wins": row.idxmax() == g, "and is above 1": row[g] > 1.0})
    return pd.DataFrame(rows).set_index("family")


def blind_battery(mods, M, va, families, fam_ngen, a_lo, a_hi, n_perm=2000, seed=0):
    """Three tests for 'the panel is blind to this family', because they fail differently.

    1. ELUB span 0            -- no LR other than 1 is empirically supportable
    2. permutation on Cllr_min -- PANEL-blind: no ordering information about this family
    3. source bound N = 1     -- REGISTRY-blind, independent of detector quality

    Plus a fourth, different failure: CONFOUND-blind, meaning no common support with the
    authentic class in the crop-geometry covariate. A family can be perfectly visible to the
    panel and still have no population to compare it against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in families:
        m = mods[g]
        sub, y = _family_val(va, g)
        lr = _bounded(m, sub[Z].values)
        obs = cllr_min(lr, y)
        null = np.array([cllr_min(lr, rng.permutation(y)) for _ in range(n_perm)])
        p = (1 + int((null <= obs).sum())) / (n_perm + 1)
        lf = M.loc[(M.label == 1) & (M.generator_family == g), "logfrac"]
        support = float(((lf >= a_lo) & (lf <= a_hi)).mean())
        rows.append({"family": g, "ngen": fam_ngen[g],
                     "elub_span_dex": np.log10(m["hi"] / m["lo"]),
                     "Cllr_min": obs, "null_median": float(np.median(null)), "perm_p": p,
                     "panel_blind (p>0.05)": p > 0.05,
                     "registry_blind (N=1)": source_bound(fam_ngen[g]) == (1.0, 1.0),
                     "confound_support": support, "confound_blind": support == 0.0})
    return pd.DataFrame(rows).set_index("family")


def multiplicity(agg, avail, dates, yv, n_boot=2000, seed=0):
    """E[max | authentic] by date, against the Bonferroni bound |G<=T|.

    A properly calibrated LR has E[LR | Hd] = 1; the gap is what taking a maximum over
    correlated family comparisons costs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t, d in enumerate(dates):
        mm = agg["max"][yv == 0, t]
        boot = np.array([mm[rng.integers(0, len(mm), len(mm))].mean() for _ in range(n_boot)])
        rows.append({"T": str(d)[:10], "|G<=T|": int(avail[t].sum()), "E[max|Hd]": mm.mean(),
                     "lo95": np.percentile(boot, 2.5), "hi95": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def tstar_table(curve, dates, thetas, true_dates, yv):
    """Crossing rates, median T*, and the temporal localization error. Returns (table, {theta: T*})."""
    tstar, rows = {}, []
    for th in thetas:
        ts = crossing(curve, dates, th)
        tstar[th] = ts
        fk = ts[yv == 1].reset_index(drop=True)
        err = localization_error(fk, true_dates)
        rows.append({"theta": th,
                     "cross rate: fake": fk.notna().mean(),
                     "authentic": ts[yv == 0].notna().mean(),
                     "median T*(fake)": str(fk.dropna().median())[:10],
                     "median T*-d_g (days)": np.median(err),
                     "share early": float(np.mean(err < 0)),
                     "median |T*-d_g| (yr)": np.median(np.abs(err)) / 365.25})
    return pd.DataFrame(rows).set_index("theta"), tstar


def calibrator_comparison(tr, va, families, l2s):
    """PAV against logistic calibration on the same per-family fused score."""
    rows = []
    for g in families:
        A, F = family_xy(tr, g)
        fused, _ = fuse(A, F, l2s[g])
        sub, y = _family_val(va, g)
        y_tr = np.r_[np.zeros(len(A)), np.ones(len(F))]
        out = {"family": g, "n_tr": len(F)}
        for tag, cal in (("PAV", pav_lr(fused(A), fused(F))),
                         ("logistic", logistic_lr(fused(A), fused(F)))):
            lo, hi = elub(cal(fused(np.vstack([A, F]))), y_tr)
            c, _, ccal = cllr_decomp(np.clip(cal(fused(sub[Z].values)), lo, hi), y)
            out[f"{tag}_Cllr"], out[f"{tag}_cal"] = c, ccal
            out[f"{tag}_span"] = np.log10(hi / lo)
        rows.append(out)
    return pd.DataFrame(rows).set_index("family")


def crossfit_table(Mi, oof, families, famtab):
    rows = []
    for j, g in enumerate(families):
        sel = ((Mi.label == 0) | (Mi.generator_family == g)).values
        y = (Mi.generator_family[sel] == g).astype(int).values
        lo, hi = elub(oof[sel, j], y)
        c, cmin, ccal = cllr_decomp(np.clip(oof[sel, j], lo, hi), y)
        rows.append({"family": g, "n_fake_oof": int(y.sum()), "Cllr_oof": c,
                     "Cllr_min_oof": cmin, "Cllr_cal_oof": ccal,
                     "elub_span": np.log10(hi / lo),
                     "Cllr_split": famtab.loc[g, "Cllr"],
                     "Cllr_cal_split": famtab.loc[g, "Cllr_cal"]})
    return pd.DataFrame(rows).set_index("family")


def residualization_slopes(tr):
    """Why residualizing scores on the covariate is a trap, not a fix.

    Fit z ~ a + b*log frac on AUTHENTIC rows; the residual is z - a - b*log frac. Where b < 0
    the residual is z + |b|*log frac - a, a two-feature classifier that adds the confound back
    with the sign that helps. Removing a covariate effect estimated within one class does not
    de-confound when the covariate distributions differ across classes; it re-confounds.
    """
    auth = tr[tr.label == 0]
    rows = []
    for d in Z:
        b, _ = np.polyfit(auth.logfrac, auth[d], 1)
        rows.append({"detector": d, "slope b on log10 frac": b,
                     "rho": np.corrcoef(auth.logfrac, auth[d])[0, 1],
                     "residual re-adds frac with sign":
                         "+ (helps separation)" if b < 0 else "- (hurts)"})
    return pd.DataFrame(rows).set_index("detector")


def summary_table(mods, tr, families, fam_ngen, fam_date, blind, famtab, floor, crossfit):
    n_auth_tr = int((tr.label == 0).sum())
    return pd.DataFrame({
        "n_gen": pd.Series(fam_ngen),
        "availability": pd.Series({g: str(fam_date[g])[:10] for g in families}),
        "panel_blind": blind["panel_blind (p>0.05)"],
        "registry_blind": blind["registry_blind (N=1)"],
        "confound_blind": blind["confound_blind"],
        "ELUB span (dex)": blind["elub_span_dex"],
        "max supportable LR": pd.Series({g: adopt_bounds(
            (mods[g]["lo"], mods[g]["hi"]),
            count_bound(mods[g]["n_tr"], n_auth_tr),
            source_bound(fam_ngen[g]))[1] for g in families}),
        "Cllr (split)": famtab["Cllr"],
        "Cllr (cross-fit)": crossfit["Cllr_oof"],
        "Cllr floor (geometry only)": floor["Cllr"],
    }).loc[families]
