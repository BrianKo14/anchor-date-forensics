"""Figures. Each function takes what it needs and returns nothing; the notebook shows them.

House style follows merge_scores.ipynb cell 1. One convention is load-bearing rather than
cosmetic: PAV yields only a handful of distinct LR values, so every LR curve here is a
staircase drawn with steps-post and a marker at each step. Never smooth them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

from .bounds import source_bound
from .metrics import det_points, tippett
from .models import lr_matrix
from .aggregate import localization_error

__all__ = ["style", "REAL", "FAKE", "MUTED", "NULLBAND", "famcolor",
           "covariate_ecdf", "fusion_heatmaps", "calibrator_cost", "tippett_grid",
           "det_and_strip", "aggregator_panel", "tstar_panel"]

REAL, FAKE = "#3b6ea5", "#c4462c"
NULLBAND, MUTED = "#d9d9d9", "#888888"
# note: famcolor["diffusion"] equals REAL, so any figure mixing authentic rows with the
# diffusion family must separate them by line style, not colour.
famcolor = {"gan": "#c4462c", "diffusion": "#3b6ea5", "inpainting": "#7a9a3b",
            "autoregressive": "#b58a2c", "graphics": "#8a6bab", "other": "#888888",
            "authentic": "#3b6ea5"}


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titleweight": "bold",
    })


def covariate_ecdf(M, families, a_lo, a_hi):
    """ECDF of log10 crop-area fraction per fake family and per authentic source."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for g in families:
        v = np.sort(M.loc[(M.label == 1) & (M.generator_family == g), "logfrac"].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color=famcolor[g], lw=1.6, label=f"{g} (n={len(v)})")
    for src, ls in zip(["COCO2017", "LAION-400M", "RAISE"], ["-", "--", ":"]):
        v = np.sort(M.loc[(M.label == 0) & (M.source == src), "logfrac"].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color="black", ls=ls, lw=1.2, alpha=0.75, label=f"authentic: {src}")
    ax.axvspan(a_lo, a_hi, color=NULLBAND, alpha=0.45, zorder=0)
    ax.set_xlabel("log10 crop-area fraction  =  log10( 200x200 / (W x H) )")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle("the covariate the LR cannot condition on   (grey = authentic 1-99 pct)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    plt.show()


def fusion_heatmaps(mods, va, families):
    """Pairwise rank correlation of the six LR_g under each fusion -- the degeneracy, visible."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    short = [g[:6] for g in families]
    for ax, (mode, mm) in zip(axes, mods.items()):
        L = lr_matrix(mm, va, families)
        R = np.corrcoef(np.apply_along_axis(rankdata, 0, L), rowvar=False)
        im = ax.imshow(R, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(families)), short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(families)), short, fontsize=7)
        ax.grid(False)
        for i in range(len(families)):
            for j in range(len(families)):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(R[i, j]) > 0.6 else "black")
        ax.set_title(f"{mode} fusion")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman")
    fig.suptitle("pairwise rank correlation of the six LR_g   (val)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    plt.show()


def calibrator_cost(mods, families, fam_ngen, fpr_grid):
    """LR_g against the authentic false-positive rate the score implies.

    Each family has its OWN fusion, so the six fused scores are different quantities and must
    not share an x-axis. The implied authentic FPR is a common, interpretable one.
    """
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for g in families:
        m = mods[g]
        thr = np.percentile(m["s_auth"], 100.0 - fpr_grid)
        ax.step(fpr_grid, np.clip(m["cal"](thr), m["lo"], m["hi"]), where="post",
                color=famcolor[g], lw=1.6,
                label=f"{g} (n_tr={m['n_tr']}, N={fam_ngen[g]})")
        ax.axhline(m["hi"], color=famcolor[g], ls=":", lw=0.7, alpha=0.55)
        ax.plot([fpr_grid[0]], [source_bound(fam_ngen[g])[1]], marker="_", ms=11, mew=2.0,
                color=famcolor[g])
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("authentic false-positive rate implied by the score "
                  "(% of authentic train at or above)")
    ax.set_ylabel("LR_g  (ELUB-bounded)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.suptitle("what each family can report, against what it costs on authentic images\n"
                 "dotted = ELUB ceiling; the tick at the left edge = the source-count ceiling",
                 x=0.005, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    plt.show()


def tippett_grid(mods, va, families, Z, famtab):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6))
    for ax, g in zip(axes.flat, families):
        m = mods[g]
        sub = va[(va.label == 0) | (va.generator_family == g)]
        y = (sub.generator_family == g).astype(int).values
        lr = np.clip(m["model"](sub[Z].values), m["lo"], m["hi"])
        x, tp, td = tippett(lr, y)
        ax.step(x, tp, where="post", color=FAKE, lw=1.6, marker="|", ms=3,
                label=f"fake (n={y.sum()})")
        ax.step(x, td, where="post", color=REAL, lw=1.6, marker="|", ms=3,
                label=f"authentic (n={(y == 0).sum()})")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{g}   Cllr={famtab.loc[g, 'Cllr']:.3f}")
        ax.set_xlabel("log10 LR")
        ax.set_ylabel("proportion with LR >= x")
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Tippett plots per family   (val, ELUB-bounded; staircases, not curves)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    plt.show()


def det_and_strip(mods, va, families, Z, rme, seed=0, min_fake=20):
    """DET only where n_val_fake >= min_fake; every individual LR for the families below it.

    At n=9 the H_p axis moves in 11% jumps, so a five-point "curve" would be a lie of
    presentation. Those families get their nine dots shown instead.
    """
    big = [g for g in families if rme.loc[g, "n_val_fake"] >= min_fake]
    small = [g for g in families if g not in big]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ticks = np.array([0.01, 0.05, 0.2, 0.6])
    for g in big:
        m = mods[g]
        sub = va[(va.label == 0) | (va.generator_family == g)]
        y = (sub.generator_family == g).astype(int).values
        fp, fn = det_points(np.clip(m["model"](sub[Z].values), m["lo"], m["hi"]), y)
        axes[0].plot(fp, fn, color=famcolor[g], lw=1.6, marker="o", ms=2.5,
                     label=f"{g} (n={y.sum()})")
    lim = norm.ppf([0.008, 0.65])
    axes[0].plot(lim, lim, color=MUTED, lw=0.6, ls="--", zorder=0)   # equal-error diagonal
    axes[0].set_xticks(norm.ppf(ticks), [f"{t:.0%}" for t in ticks], fontsize=7)
    axes[0].set_yticks(norm.ppf(ticks), [f"{t:.0%}" for t in ticks], fontsize=7)
    axes[0].set_xlabel("false positive rate (authentic called fake)")
    axes[0].set_ylabel("false negative rate")
    axes[0].set_title(f"DET -- only where n_val_fake >= {min_fake}")
    axes[0].legend(frameon=False, fontsize=7)

    for i, g in enumerate(small):
        m = mods[g]
        sub = va[(va.label == 0) | (va.generator_family == g)]
        y = (sub.generator_family == g).astype(int).values
        lr = np.clip(m["model"](sub[Z].values), m["lo"], m["hi"])
        jit = (np.random.default_rng(seed + i).random((y == 1).sum()) - 0.5) * 0.16
        axes[1].scatter(np.log10(lr[y == 1]), np.full((y == 1).sum(), i) + jit,
                        s=22, color=famcolor[g], zorder=3, label=f"{g} (n={y.sum()})")
        q = np.percentile(np.log10(lr[y == 0]), [5, 50, 95])
        axes[1].plot(q[[0, 2]], [i - 0.26] * 2, color=REAL, lw=2.5, alpha=0.75,
                     solid_capstyle="butt")
        axes[1].plot([q[1]], [i - 0.26], marker="|", color=REAL, ms=9)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(range(len(small)), small, fontsize=7)
    axes[1].set_xlabel("log10 LR   (dots = each individual fake; blue bar = authentic 5-50-95 pct)")
    axes[1].set_title("every fake, individually -- n is too small for a curve")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left",
                   bbox_to_anchor=(0.0, 1.16), ncol=3)
    fig.tight_layout()
    plt.show()


def aggregator_panel(agg, mult, va, dates, thetas, yv):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))

    ax = axes[0]
    ax.plot(mult["T"], mult["|G<=T|"], color=MUTED, ls="--", marker="s", ms=4,
            label="Bonferroni bound  |G<=T|")
    ax.fill_between(mult["T"], mult.lo95, mult.hi95, color=FAKE, alpha=0.18)
    ax.plot(mult["T"], mult["E[max|Hd]"], color=FAKE, marker="o", ms=4,
            label="measured E[max | authentic]")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_ylabel("expected LR on an authentic image")
    ax.set_title("multiplicity: what the max costs on authentic images")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.legend(frameon=False, fontsize=7)

    # famcolor["diffusion"] is the same blue as REAL, so authentic rows get their own styles
    ax = axes[1]
    picks = []
    for src, ls in (("RAISE", "--"), ("COCO2017", ":")):
        picks.append((va[(va.label == 0) & (va.source == src)].index[0],
                      f"authentic / {src}", "black", ls))
    for g in ["gan", "diffusion", "graphics"]:
        sel = va[(va.label == 1) & (va.generator_family == g)]
        picks.append((sel.index[0], f"{g} / {sel.generator.iloc[0]}", famcolor[g], "-"))
    pos = {ix: i for i, ix in enumerate(va.index)}
    xs = np.arange(len(dates))
    for ix, tag, col, ls in picks:
        ax.step(xs, agg["max"][pos[ix]], where="post", color=col, ls=ls, lw=1.7,
                marker="o", ms=3, label=tag)
    for th in thetas:
        ax.axhline(th, color=MUTED, ls=":", lw=0.8)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(xs, [str(d)[:10] for d in dates], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("LR(F, T)   (log scale)")
    ax.set_title("LR(F,T) for five example val images")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    axr = ax.twinx()          # thresholds labelled clear of the curves and the legend
    axr.set_yscale("log")
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks(list(thetas), [f"θ={t:g}" for t in thetas], fontsize=6.5)
    axr.tick_params(length=0, colors=MUTED)
    axr.minorticks_off()
    axr.grid(False)

    fig.suptitle("the aggregator over G<=T   (val, per-family fusion, ELUB-bounded)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    plt.show()


def tstar_panel(tstar, dates, thetas, yv, true_dates):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    xs = np.arange(len(dates))
    w = 0.26

    ax = axes[0]
    for k, th in enumerate(thetas):
        ts = tstar[th]
        cf = np.array([(ts[yv == 1].dt.date == pd.Timestamp(d).date()).sum() for d in dates])
        ca = np.array([(ts[yv == 0].dt.date == pd.Timestamp(d).date()).sum() for d in dates])
        ax.bar(xs + (k - 1) * w, cf, w * 0.92, color=FAKE, alpha=0.35 + 0.3 * k,
               label=f"fake, θ={th:g}")
        ax.bar(xs + (k - 1) * w, -ca, w * 0.92, color=REAL, alpha=0.35 + 0.3 * k,
               label=f"authentic, θ={th:g}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs, [str(d)[:10] for d in dates], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("images crossing at this T  (fake up / authentic down)")
    ax.set_title("where T*(F) lands")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[1]
    for k, th in enumerate(thetas):
        err = localization_error(tstar[th][yv == 1], true_dates) / 365.25
        v = np.sort(err)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                lw=1.7, color=plt.cm.viridis(k / 3), label=f"θ={th:g} (n={len(v)})")
    ax.axvline(0, color="black", lw=1.0)
    ax.axvspan(-8, 0, color=FAKE, alpha=0.07, zorder=0)
    ax.text(-0.15, 0.06, "early: says fabrication was plausible\nbefore the generator existed",
            fontsize=6.5, color=FAKE, ha="right")
    ax.set_xlabel("temporal localization error  T* - d_g  (years)")
    ax.set_ylabel("ECDF")
    ax.set_title("the error is overwhelmingly early")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("T*(F): pricing the timestamp   (val, family granularity)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    plt.show()
