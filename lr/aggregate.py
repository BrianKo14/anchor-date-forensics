"""Aggregating per-family LRs over G<=T, and the crossing point T*(F).

On the primary aggregator: max_g LR_g = sup over the within-prosecution prior of a mixture, so
it is a PROFILE likelihood ratio over a composite hypothesis. That makes it conservative
*against missing a family* but anti-conservative *as evidence of fabrication*, and therefore
anti-conservative for T*. Those are opposite senses of "conservative"; CLAUDE.md's wording
covers only the first.
"""

import numpy as np
import pandas as pd

__all__ = ["build_aggregators", "monotone_violations", "crossing", "localization_error"]


def build_aggregators(L, dates, key_date, keys):
    """dict of name -> (n x |dates|) LR curve, plus the |dates| x |keys| availability mask.

    `max` is monotone in T by a one-line argument: the max of a FIXED set of reals over a
    NESTED increasing family of index sets is non-decreasing. Both hypotheses are load-bearing
    -- the /|G<=T| variants break the first and truncated models break the second -- so the
    caller should assert monotonicity rather than assume it.
    """
    avail = np.array([[key_date[g] <= T for g in keys] for T in dates])
    n_av = avail.sum(1)
    mx = np.stack([np.where(avail[t], L, -np.inf).max(1) for t in range(len(dates))], axis=1)
    mix = np.stack([np.where(avail[t], L, 0.0).sum(1) / n_av[t] for t in range(len(dates))],
                   axis=1)
    return {"max": mx,
            "max / |G|": mx / len(keys),          # constant divisor: still monotone
            "max / |G<=T|": mx / n_av[None, :],   # T-dependent divisor: expected to fail
            "uniform mixture over G<=T": mix}, avail


def monotone_violations(curve, tol=1e-12):
    """Count of date transitions where the curve decreases."""
    return int((np.diff(curve, axis=1) < -tol).sum())


def crossing(curve, dates, theta):
    """T*(F): the earliest grid date at which a curve reaches theta; NaT if it never does.

    argmax over a BOOLEAN array, not idxmax over floats, so tied dates return the earliest.
    """
    hit = curve >= theta
    out = pd.Series(pd.to_datetime(dates[hit.argmax(1)]))
    return out.where(pd.Series(hit.any(1)), pd.NaT)


def localization_error(tstar, true_dates):
    """T* - d_g in days, over the images that crossed. Negative == early, the dangerous side."""
    tstar = tstar.reset_index(drop=True)
    true_dates = pd.Series(np.asarray(true_dates)).reset_index(drop=True)
    ok = tstar.notna().values
    return (tstar[ok].values - true_dates[ok].values) / np.timedelta64(1, "D")
