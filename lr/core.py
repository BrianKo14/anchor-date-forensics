"""The 50/50 construction: calibrators and fusion that emit likelihood ratios directly.

Every estimator here is fitted with each class carrying total weight 1/2, so the prior odds
are 1 and posterior odds ARE the likelihood ratio -- no prevalence assumption enters. See
lr/README.md for the algebra.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["balanced_weights", "pav_lr", "logistic_lr", "fuse", "auroc"]


def balanced_weights(y):
    """Weights putting total mass 1/2 on each class, so posterior odds == LR exactly."""
    y = np.asarray(y)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    assert n1 > 0 and n0 > 0, "both classes must be present"
    return np.where(y == 1, 0.5 / n1, 0.5 / n0)


def pav_lr(s_auth, s_fake):
    """Pool-adjacent-violators / isotonic calibrator. Returns a score -> LR function.

    Discriminative: this fits the LR function directly, score -> probability -> LR. It never
    estimates the two class densities separately and divides them.

    out_of_bounds="clip" matters: a validation score outside the training range would otherwise
    predict NaN. Clipping to the terminal block is the only defensible extrapolation, and ELUB
    then bounds whatever that block claims.
    """
    s = np.r_[np.asarray(s_auth, float), np.asarray(s_fake, float)]
    y = np.r_[np.zeros(len(s_auth)), np.ones(len(s_fake))]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y, sample_weight=balanced_weights(y))

    def to_lr(x):
        p = iso.predict(np.asarray(x, float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return p / (1.0 - p)
    return to_lr


def logistic_lr(s_auth, s_fake):
    """Logistic calibrator, the comparator to PAV."""
    s = np.r_[np.asarray(s_auth, float), np.asarray(s_fake, float)].reshape(-1, 1)
    y = np.r_[np.zeros(len(s_auth)), np.ones(len(s_fake))]
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(s, y, sample_weight=balanced_weights(y) * len(y))
    return lambda x: np.exp(clf.decision_function(np.asarray(x, float).reshape(-1, 1)))


def fuse(X_auth, X_fake, l2):
    """Ridge logistic fusion of a detector-score matrix into one scalar. Returns (X -> score, clf).

    Weights sum to n rather than 1 so l2 means the same thing across families of very different
    size -- sklearn does not rescale the penalty by the sample weights, so without this a given
    lambda would regularise a 137-fake family and an 8-fake family differently.
    """
    X = np.vstack([np.asarray(X_auth, float), np.asarray(X_fake, float)])
    y = np.r_[np.zeros(len(X_auth)), np.ones(len(X_fake))]
    clf = LogisticRegression(C=1.0 / l2, l1_ratio=0, solver="lbfgs", max_iter=10000)
    clf.fit(X, y, sample_weight=balanced_weights(y) * len(y))
    return (lambda P: clf.decision_function(np.asarray(P, float))), clf


def auroc(score, y):
    """Rank-based AUROC, matching merge_scores.ipynb section 5.

    Reported only for continuity with the existing notebooks. AUROC is the wrong statistic for
    this problem (CLAUDE.md): it is driven by the bottom of the fake-score distribution, where
    this system needs upper-tail discrimination.
    """
    score, y = np.asarray(score, float), np.asarray(y)
    r = rankdata(score)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
