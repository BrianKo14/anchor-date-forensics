"""ELUB and its cross-checks: how far a finite sample lets you state a likelihood ratio.

Bounding is not a taste parameter here. It is derived from a criterion every genuine LR must
satisfy -- Markov's inequality applied to the LR itself gives Royall's bound

    P(LR >= tau | Hd) = int_{LR>=tau} f_d = int_{LR>=tau} f_p/LR <= (1/tau) int f_p <= 1/tau

and ELUB is the empirical enforcement of exactly that, expressed as a normalised Bayes error
rate. See lr/README.md.
"""

import numpy as np

__all__ = ["nbe_max", "elub", "count_bound", "source_bound", "adopt_bounds"]


def nbe_max(lr_p, lr_d, lower, upper):
    """sup over tau of the normalised Bayes error rate, for LRs clipped to [lower, upper] and
    augmented with one maximally-misleading observation per class.

        NBE(tau) = [P(LR<=tau|Hp) + tau*P(LR>tau|Hd)] / min(1, tau)

    The added misleading pair is the devil's advocate -- "suppose the next case is the worst
    this bound permits". It is what makes the bound finite when no misleading evidence has
    actually been observed, and what makes the bound scale with sample size.
    """
    P = np.r_[np.clip(np.asarray(lr_p, float), lower, upper), lower]
    D = np.r_[np.clip(np.asarray(lr_d, float), lower, upper), upper]
    pts = np.unique(np.r_[P, D, 1.0])
    pts = pts[np.isfinite(pts) & (pts > 0)]
    # tau * P(LR>tau|Hd) rises linearly between jumps, so the sup sits at a jump or its left limit
    grid = np.unique(np.r_[pts, pts * (1 - 1e-12), pts * (1 + 1e-12), 1.0])
    grid = grid[grid > 0]
    miss = (P[None, :] <= grid[:, None]).mean(1)
    fa = (D[None, :] > grid[:, None]).mean(1)
    nbe = np.where(grid >= 1.0, miss + grid * fa, miss / grid + fa)
    return float(nbe.max())


def _widest(admissible, anchor, grid, tol=1e-6):
    """Largest x along `grid` with every point up to it admissible, refined by bisection.

    Scans rather than bisecting blind: NBE is empirically monotone in each bound, but that is an
    observation and not a theorem, so the scan asserts it instead of assuming it.
    """
    ok = np.array([admissible(x) for x in grid])
    first_bad = int(np.argmax(~ok)) if (~ok).any() else len(ok)
    assert not ok[first_bad:].any(), (
        "NBE is non-monotone in this bound: admissible points beyond the first violation")
    if first_bad == 0:
        return float(anchor)
    a = grid[first_bad - 1]
    if first_bad == len(ok):
        return float(a)
    b = grid[first_bad]
    for _ in range(60):
        if abs(np.log(b / a)) < tol:
            break
        mid = float(np.sqrt(a * b))
        a, b = (mid, b) if admissible(mid) else (a, mid)
    return float(a)


def elub(lrs, y, n_grid=200, span=1e6, rounds=4, tol=1e-4):
    """Empirical lower and upper bound: the widest (l, u) whose clipped, devil's-advocate-
    augmented LR set never does worse than the neutral system at any prior.

    A pair is admissible iff nbe_max(...) <= 1. The admissible set is a Pareto frontier, so
    this returns the fixed point of alternating maximisation, widening the upper bound first.
    (1, 1) is always admissible -- that is what a system with nothing supportable returns.

    Sanity anchor: for perfectly separated classes this saturates at count_bound(n_p, n_d),
    which is the most any finite sample can support.
    """
    lrs, y = np.asarray(lrs, float), np.asarray(y)
    lr_p, lr_d = lrs[y == 1], lrs[y == 0]
    lo, hi = 1.0, 1.0
    up_grid = np.logspace(0, np.log10(span), n_grid)[1:]
    down_grid = 1.0 / up_grid
    for _ in range(rounds):
        new_hi = _widest(lambda u: nbe_max(lr_p, lr_d, lo, u) <= 1.0, hi, up_grid)
        new_lo = _widest(lambda l: nbe_max(lr_p, lr_d, l, new_hi) <= 1.0, lo, down_grid)
        done = abs(np.log(new_hi / hi)) < tol and abs(np.log(new_lo / lo)) < tol
        lo, hi = new_lo, new_hi
        if done:
            break
    assert lo <= 1.0 <= hi, f"elub bounds must straddle 1, got ({lo}, {hi})"
    assert nbe_max(lr_p, lr_d, lo, hi) <= 1.0 + 1e-9, "elub returned an inadmissible pair"
    return lo, hi


def count_bound(n_p, n_d):
    """No LR beyond the class counts is observable: a rate below 1/n cannot be resolved."""
    return 1.0 / n_p, float(n_d)


def source_bound(n_generators):
    """[1/N, N] with N the number of generators in the family.

    The unit of analysis is the generator, not the image -- the rule temporal_correlation.ipynb
    section 5 established for this repo (272 diffusion images are 16 observations). No source
    can provide more than N-fold support for one hypothesis over the other, so a family with a
    single generator returns [1, 1]: nothing supportable, at any detector quality.
    """
    return 1.0 / n_generators, float(n_generators)


def adopt_bounds(*bounds):
    """The tightest of the bounds offered."""
    return max(b[0] for b in bounds), min(b[1] for b in bounds)
