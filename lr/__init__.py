"""Time-calibrated likelihood ratios for AI-image detection.

The layers CLAUDE.md specifies, as importable pieces so the notebook can stay a narrative:

    data       load the scored panel, audit its provenance, rescale without leakage
    core       the 50/50 construction -- PAV / logistic calibrators, ridge fusion
    metrics    Cllr and its decomposition, Tippett, DET, rates of misleading evidence
    bounds     ELUB, plus the count and source-count cross-checks
    models     per-family score models (fusion -> calibration -> bounds) and cross-fitting
    aggregate  max over G<=T, monotonicity checks, the crossing point T*(F)
    plots      the figures
    selftest   controls the rest of the pipeline depends on

Two design points are correctness requirements rather than preferences, and both are argued in
the module docstrings: fusion must be per-family (models.py), and LRs must be bounded before
Cllr is reported (metrics.check_pav_identity).
"""

from . import aggregate, bounds, core, data, metrics, models, plots, report, selftest
from .aggregate import build_aggregators, crossing, localization_error, monotone_violations
from .bounds import adopt_bounds, count_bound, elub, nbe_max, source_bound
from .core import auroc, balanced_weights, fuse, logistic_lr, pav_lr
from .data import (DETECTORS, SCORED_COLUMNS, Z, family_registry, generator_registry, kfolds,
                   load_scored_manifest, provenance, sigma_rescale, zcols)
from .metrics import (check_pav_identity, cllr, cllr_decomp, cllr_min, det_points,
                      rates_misleading, tippett)
from .models import (L2_GRID, cross_fit, family_xy, fit_all, fit_family, fit_generator_models,
                     fit_truncated_models, lr_matrix, pick_l2)

__all__ = [n for n in dir() if not n.startswith("_")]
