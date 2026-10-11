"""The forensic residual extractor (FRE): K constrained linear predictors, learned on real images.

Paper sec. 3.1. Each filter w_k is M x M with a zero centre and off-centre weights summing to 1
(upstream fsd/fre.py, ConstrainedConv2d), and predicts a pixel from its neighbourhood. The
filters minimise

    L(w) = L_E(w) + lambda * L_div(w)
    L_E   = sum_k mean_{x,y} (I(x,y) - w_k . N(x,y))^2          prediction-error energy
    L_div = -sum_i log(sigma_i(W) + alpha)                       spectral diversity

where N(x,y) is the M*M - 1 off-centre neighbourhood and W stacks the vec(w_k) as rows. Without
L_div the problem is convex with one minimiser, so all K filters would collapse onto the same
predictor; L_div is what makes K > 1 mean anything.

Why no torch and no data loader: for linear filters, L_E depends on the training images only
through the neighbourhood second-moment matrix R = mean N_full N_full^T (M^2 x M^2, centre
included). One pass over the real images accumulates R; after that every gradient step is a
(K x M^2) x (M^2 x M^2) product, so full-batch optimisation to convergence takes seconds. The
paper's "AdamW, lr 1e-3, 10 epochs" minibatch run approximates the same objective.

Assumptions the paper leaves open, recorded in DECISIONS.md D-006 and in every FRE's sidecar:
  * lambda. The paper says lambda = 1 against a loss written as a sum over pixels, without
    saying how it is normalised, and that number does not transfer: with L_E as a per-pixel
    mean, lambda = 1 leaves the filters nearly identical (minor singular values ~0.13), while the
    authors' released K=8 FRE has minor singular values ~0.95 and pays ~22% extra residual
    energy for them. So lambda is *calibrated*, not guessed: `calibrate_lambda` bisects for the
    lambda at which our K=8, M=15 FRE, trained on our reals, matches the released FRE's mean
    minor singular value (UPSTREAM_SINGULAR_VALUES). That lambda is then held fixed for every K
    and M, as the paper holds its lambda fixed.
  * alpha (the paper's "small constant") = 1e-3.
  * M = 11 as the paper states; the authors' released weights use 15.
  * Pixels are luma in 0..255, as in upstream's inference code.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import load_gray

__all__ = ["second_moments", "train", "calibrate_lambda", "save", "load", "fre_name",
           "fetch_upstream"]

# The authors' released FRE (GitHub release v1.2.0, fre.pt), decoded by fetch_upstream: its
# constrained filters' singular values, and the file's hash. The calibration target for lambda.
UPSTREAM_URL = ("https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/"
                "releases/download/v1.2.0/fre.pt")
UPSTREAM_SHA256 = "d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb"
UPSTREAM_SINGULAR_VALUES = (2.636909, 0.996043, 0.994293, 0.99183,
                            0.94946, 0.938894, 0.893224, 0.889896)


def fre_name(K, M):
    return f"fre_K{K}_M{M}"


def second_moments(paths, M):
    """R = mean over every valid M x M window of every image of vec(window) vec(window)^T."""
    R = np.zeros((M * M, M * M))
    n = 0
    for path in paths:
        windows = sliding_window_view(load_gray(path), (M, M)).reshape(-1, M * M)
        R += windows.T @ windows
        n += len(windows)
    return R / n, n


def _split(R, M):
    """R partitioned into centre/neighbourhood blocks: (r_cc, r_nc, R_nn)."""
    c = (M * M) // 2
    nb = np.r_[0:c, c + 1:M * M]
    return R[c, c], R[nb, c], R[np.ix_(nb, nb)]


def _embed(V, M):
    """(K, M*M - 1) off-centre weights -> (K, M, M) filters with a zero centre."""
    c = (M * M) // 2
    full = np.insert(V, c, 0.0, axis=1)
    return full.reshape(len(V), M, M)


def loss_terms(V, r_cc, r_nc, R_nn, alpha):
    energy = r_cc - 2 * V @ r_nc + np.einsum("ki,ij,kj->k", V, R_nn, V)
    sigma = np.linalg.svd(V, compute_uv=False)
    return energy, -np.log(sigma + alpha).sum(), sigma


def train(R, M, K, lam=1.0, alpha=1e-3, seed=0, init=None, max_iter=50_000):
    """Minimise L_E + lam * L_div. Returns (filters (K,M,M), info dict).

    Constraint and conditioning are both removed by a change of variables. Each filter's
    off-centre weights are v_k = 1/n + P z_k, with P an orthonormal basis of the sum-zero
    subspace, so sum(v_k) = 1 holds for every z. The energy is then a quadratic in z with Hessian
    H = 2 P^T R_nn P, whose eigenvalues span many decades on natural images (plain gradient
    methods, Adam included, stall on it). Substituting z_k = C^{-T} u_k with H/2 = C C^T makes
    the energy isotropic in u, and L-BFGS converges in a few hundred iterations. With lam = 0
    the result equals the closed-form constrained least-squares predictor (selftest checks it).
    """
    from scipy.optimize import minimize

    n = M * M - 1
    r_cc, r_nc, R_nn = _split(R, M)
    v0 = np.full(n, 1.0 / n)
    P = np.linalg.svd(np.ones((1, n)))[2][1:].T                  # (n, n-1), orthonormal, sum-zero
    C = np.linalg.cholesky(P.T @ R_nn @ P)
    Cinv_T = np.linalg.inv(C).T

    def to_V(u):
        return v0 + (u.reshape(K, n - 1) @ Cinv_T.T) @ P.T

    def objective(u):
        V = to_V(u)
        energy = r_cc - 2 * V @ r_nc + np.einsum("ki,ij,kj->k", V, R_nn, V)
        grad_V = 2 * (V @ R_nn - r_nc[None])
        total = energy.sum()
        if lam:
            U, s, Vt = np.linalg.svd(V, full_matrices=False)
            total -= lam * np.log(s + alpha).sum()
            grad_V -= lam * (U / (s + alpha)) @ Vt
        grad_u = (grad_V @ P) @ Cinv_T
        return total, grad_u.ravel()

    rng = np.random.default_rng(seed)
    if init is None:
        # upstream's xavier_normal(gain=1/3) scale, projected onto the constraint
        V_init = v0 + rng.normal(0.0, 1.0 / 3.0 * np.sqrt(2.0 / (2 * n)), size=(K, n))
    else:
        V_init = np.array(init, dtype=np.float64).reshape(K, n)
    z_init = (V_init - v0) @ P                                      # project onto the constraint
    u_init = z_init @ C                                             # z = C^{-T} u  <=>  u = z C
    result = minimize(objective, u_init.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "maxfun": 2 * max_iter,
                               "ftol": 1e-15, "gtol": 1e-10})
    V = to_V(result.x)
    # L-BFGS-B often ends on "ABNORMAL" here: at ftol=1e-15 the line search runs out of float
    # precision before the gradient test fires. The whitened gradient norm is the honest check
    # (in these coordinates the energy Hessian is the identity, so it bounds the distance to the
    # optimum directly); two seeds agree on the loss to ~1e-13 relative on the sample.
    grad_norm = float(np.linalg.norm(objective(result.x)[1]))

    energy, div, sigma = loss_terms(V, r_cc, r_nc, R_nn, alpha)
    info = {
        "K": K, "M": M, "lambda": lam, "alpha": alpha, "seed": seed,
        "optimizer": "L-BFGS-B, whitened sum-zero coordinates",
        "iterations": int(result.nit), "grad_norm_whitened": grad_norm,
        "converged": bool(result.success or grad_norm < 1e-4),
        "optimizer_message": str(result.message),
        "energy_per_filter": energy.tolist(), "diversity": float(div),
        "loss": float(energy.sum() + lam * div), "singular_values": sigma.tolist(),
    }
    return _embed(V, M), info


def calibrate_lambda(R15, target=UPSTREAM_SINGULAR_VALUES, lo=1.0, hi=500.0, rtol=1e-3):
    """lambda at which a K=8, M=15 FRE trained on R15 matches `target`'s mean minor singular value.

    Mean minor singular value (all but the largest) is monotone increasing in lambda -- more
    weight on diversity spreads the spectrum -- so plain bisection on a log scale suffices.
    The largest singular value is excluded because it mostly reflects the shared low-pass
    component every predictor has, not diversity between filters.
    """
    goal = float(np.mean(target[1:]))
    K = len(target)

    def minor(lam):
        _, info = train(R15, 15, K, lam=lam)
        return float(np.mean(info["singular_values"][1:]))

    assert minor(lo) < goal < minor(hi), "calibration bracket does not contain the target"
    while hi / lo > 1 + rtol:
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if minor(mid) < goal else (lo, mid)
    lam = float(np.sqrt(lo * hi))
    return lam, {"target_mean_minor_sv": goal, "achieved_mean_minor_sv": minor(lam)}


def fetch_upstream(dest_dir):
    """Download the authors' released FRE and decode it to the same (K, M, M) layout as ours.

    No torch needed: a torch .pt is a zip whose data/<n> members are raw little-endian storage.
    data/0 is the unconstrained weight (8, 1, 15, 15); the constraint is upstream's
    ConstrainedConv2d.constrain, re-applied here. Checked against UPSTREAM_SHA256 and
    UPSTREAM_SINGULAR_VALUES, so a changed release cannot slip in.
    """
    import io
    import urllib.request
    import zipfile

    raw = urllib.request.urlopen(urllib.request.Request(
        UPSTREAM_URL, headers={"User-Agent": "anchor-date-forensics"}), timeout=120).read()
    assert hashlib.sha256(raw).hexdigest() == UPSTREAM_SHA256, "upstream fre.pt changed"
    archive = zipfile.ZipFile(io.BytesIO(raw))
    assert archive.read("fre/byteorder") == b"little"
    w = np.frombuffer(archive.read("fre/data/0"), dtype="<f4").astype(np.float64)
    K, M = 8, 15
    w = w.reshape(K, M * M)
    centre = np.zeros(M * M)
    centre[(M * M) // 2] = 1.0
    w = w - w.mean(axis=1, keepdims=True) + 1.0 / (M * M - 1)
    w = w / (w * (1 - centre)).sum(axis=1, keepdims=True)
    w = w - w * centre
    filters = w.reshape(K, M, M)
    sigma = np.linalg.svd(w, compute_uv=False)
    assert np.allclose(sigma, UPSTREAM_SINGULAR_VALUES, atol=1e-5)

    info = {"K": K, "M": M, "source": UPSTREAM_URL, "source_sha256": UPSTREAM_SHA256,
            "license": "CC BY-NC-SA 4.0 (upstream); not redistributed, fetched on demand",
            "singular_values": sigma.tolist()}
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"fre_upstream_K{K}_M{M}.npz"
    np.savez(path, filters=filters)
    (dest_dir / f"fre_upstream_K{K}_M{M}.meta.json").write_text(
        json.dumps(dict(info, sha256=filters_sha256(filters)), indent=2) + "\n")
    return path


def save(filters, info, out_dir, extra=None):
    """Write <name>.npz (the filters) and <name>.meta.json (how they were made, and their hash)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K, M, _ = filters.shape
    name = fre_name(K, M)
    np.savez(out_dir / f"{name}.npz", filters=filters)
    meta = dict(info, **(extra or {}), sha256=filters_sha256(filters))
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir / f"{name}.npz"


def filters_sha256(filters):
    """Hash of the filter values themselves, not the .npz container (zip timestamps vary)."""
    return hashlib.sha256(np.ascontiguousarray(filters, dtype=np.float64).tobytes()).hexdigest()


def load(path):
    """(filters, sha256). Asserts the constraints, so a corrupted file cannot pass silently."""
    filters = np.load(path)["filters"]
    K, M, _ = filters.shape
    assert np.all(filters[:, M // 2, M // 2] == 0), f"{path}: centre weight not zero"
    assert np.allclose(filters.reshape(K, -1).sum(axis=1), 1.0, atol=1e-9), \
        f"{path}: filters do not sum to 1"
    return filters, filters_sha256(filters)
