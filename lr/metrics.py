"""The forensic-science evaluation battery: Cllr and its decomposition, Tippett, DET, RME."""

import numpy as np
from scipy.stats import rankdata, norm

from .core import pav_lr

__all__ = ["cllr", "cllr_min", "cllr_decomp", "check_pav_identity",
           "rates_misleading", "tippett", "det_points"]


def cllr(lrs, y):
    """Brummer & du Preez log-likelihood-ratio cost. 1.0 == the neutral system (LR == 1).

    Infinite whenever an authentic case carries LR = inf or a fake case LR = 0, which is why
    LRs must be ELUB-bounded before this is reported. Cllr on unbounded out-of-sample PAV
    output is routinely inf.
    """
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.log2(1.0 + 1.0 / lrs[y == 1])    # LR=0 -> inf, LR=inf -> 0
        term_d = np.log2(1.0 + lrs[y == 0])          # LR=inf -> inf
    return 0.5 * (term_p.mean() + term_d.mean())


def cllr_min(lrs, y):
    """Discrimination component: Cllr after optimal monotone (PAV) recalibration.

    Depends only on the ORDERING of lrs, so it is computed on ranks -- which also makes it
    immune to the +-inf atoms unbounded PAV produces. Always finite: a PAV block with p = 1
    contains no authentics, so every infinite term is multiplied by an empty average.
    """
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    r = rankdata(lrs)                                # average ranks: tied LRs share a block
    return cllr(pav_lr(r[y == 0], r[y == 1])(r), y)


def cllr_decomp(lrs, y):
    """(Cllr, Cllr_min, Cllr_cal).

    Cllr_cal is an UPPER bound on calibration loss: Cllr_min comes from PAV on the same data
    being scored, so it is optimistically biased.
    """
    c, cmin = cllr(lrs, y), cllr_min(lrs, y)
    return c, cmin, c - cmin


def check_pav_identity(s_auth, s_fake, tag=""):
    """Assert the identity that makes ELUB mandatory rather than cosmetic.

    For 50/50-weighted PAV, in-sample and exactly:
        E[LR | Hd]   == 1 - P(LR = inf | Hp)
        E[1/LR | Hp] == 1 - P(LR = 0   | Hd)
    A PAV block b holding n_pb fakes and n_db authentics gets LR_b = (n_pb/n_p)/(n_db/n_d), so
    E[LR|Hd] = sum_b (n_db/n_d) LR_b = sum over blocks WITH authentics of n_pb/n_p. Any fake
    mass in a block with no authentics escapes into an LR = inf atom and the expectation falls
    short by exactly that mass -- so E[LR|Hd] = 1, the defining property of a calibrated LR,
    fails for unbounded PAV output.
    """
    f = pav_lr(s_auth, s_fake)
    lr_d, lr_p = f(np.asarray(s_auth, float)), f(np.asarray(s_fake, float))
    with np.errstate(divide="ignore"):
        inv_p = 1.0 / lr_p
    e_d, rhs_d = lr_d.mean(), 1.0 - np.isinf(lr_p).mean()
    e_p, rhs_p = inv_p.mean(), 1.0 - (lr_d == 0).mean()
    assert abs(e_d - rhs_d) < 1e-9, f"{tag}: E[LR|Hd] identity violated"
    assert abs(e_p - rhs_p) < 1e-9, f"{tag}: E[1/LR|Hp] identity violated"
    return {"family": tag, "E[LR|Hd]": e_d, "1-P(LR=inf|Hp)": rhs_d,
            "fake mass in the inf atom": 1.0 - e_d, "E[1/LR|Hp]": e_p}


def rates_misleading(lrs, y, thresholds=(1.0, 4.0, 10.0)):
    """RME_d: authentic images reported at LR >= t. RME_p: fakes reported at LR <= 1."""
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    out = {f"RME_d(LR>={t:g})": float((lrs[y == 0] >= t).mean()) for t in thresholds}
    out["RME_p(LR<=1)"] = float((lrs[y == 1] <= 1.0).mean())
    return out


def tippett(lrs, y, n_grid=400):
    """Proportion of each class reported at LR >= x, over a log10-LR grid."""
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    fin = lrs[np.isfinite(lrs) & (lrs > 0)]
    lo = np.log10(fin.min()) - 0.3 if len(fin) else -1.0
    hi = np.log10(fin.max()) + 0.3 if len(fin) else 1.0
    x = np.linspace(lo, hi, n_grid)
    t = 10.0 ** x
    return (x,
            (lrs[y == 1][None, :] >= t[:, None]).mean(1),
            (lrs[y == 0][None, :] >= t[:, None]).mean(1))


def det_points(lrs, y):
    """(probit FPR, probit FNR) over all thresholds, for a DET plot."""
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    eps = 0.5 / max((y == 1).sum(), (y == 0).sum())
    thr = np.unique(lrs)
    thr = np.r_[thr, thr.max() * (1 + 1e-9)]
    fpr = (lrs[y == 0][None, :] >= thr[:, None]).mean(1)
    fnr = (lrs[y == 1][None, :] < thr[:, None]).mean(1)
    return norm.ppf(np.clip(fpr, eps, 1 - eps)), norm.ppf(np.clip(fnr, eps, 1 - eps))
