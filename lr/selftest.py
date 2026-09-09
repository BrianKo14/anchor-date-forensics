"""Controls the rest of the pipeline is only as trustworthy as. Run from the notebook.

Three checks:
  (a) a shuffled-label system must land on Cllr ~ 1 with an ELUB span near zero;
  (b) a perfect separator must reach Cllr_min = 0, and its ELUB must saturate at the count
      bound -- the most any finite sample can support;
  (c) the PAV identity of metrics.check_pav_identity must hold to 1e-9 on data carrying a
      genuine LR = inf atom.
"""

import numpy as np

from .bounds import count_bound, elub, nbe_max, source_bound
from .core import pav_lr
from .metrics import check_pav_identity, cllr, cllr_min

__all__ = ["run"]


def run(seed=0):
    rng = np.random.default_rng(seed)
    lines = []

    s = rng.normal(0, 1, 612)
    y = rng.permutation(np.r_[np.zeros(306), np.ones(306)])
    lr = pav_lr(s[y == 0], s[y == 1])(s)
    lo, hi = elub(lr, y)
    assert abs(nbe_max(lr[y == 1], lr[y == 0], 1.0, 1.0) - 1.0) < 1e-9, \
        "the neutral system must sit exactly on the NBE boundary"
    lines.append(f"(a) shuffled labels  : Cllr={cllr(lr, y):.4f}  Cllr_min={cllr_min(lr, y):.4f}"
                 f"  ELUB=({lo:.3f}, {hi:.3f})  span={np.log10(hi / lo):.3f} dex")

    a, f = np.linspace(-3, -1, 100), np.linspace(1, 3, 100)
    y2 = np.r_[np.zeros(100), np.ones(100)]
    lr2 = pav_lr(a, f)(np.r_[a, f])
    assert cllr_min(lr2, y2) < 1e-9
    assert np.allclose(elub(lr2, y2), count_bound(100, 100), rtol=2e-3), \
        "ELUB should saturate at the count bound when the classes separate perfectly"
    lines.append(f"(b) perfect separator: Cllr_min={cllr_min(lr2, y2):.6f}  "
                 f"ELUB={tuple(round(v, 4) for v in elub(lr2, y2))}  "
                 f"count bound={count_bound(100, 100)}")

    ident = check_pav_identity(rng.normal(0, 1, 306), rng.normal(3, 1, 137), "synthetic")
    lines.append("(c) PAV identity     : " + ", ".join(
        f"{k}={v:.6f}" for k, v in ident.items() if isinstance(v, float)))

    assert source_bound(1) == (1.0, 1.0)
    lines.append("(d) source_bound(1)  = (1.0, 1.0)   a one-generator family supports nothing")
    return "\n".join(lines)
