"""One image's forensic self-description Phi = {phi_1, ..., phi_K}.

Port of the per-image half of Nguyen, Azizpour & Stamm, "Forensic Self-Descriptions Are All You
Need..." (CVPR 2025, arXiv 2503.21003), sec. 3.2, following the authors' reference code
(github.com/ductai199x/Forensic-Self-Descriptions-CVPR25 @ 50f2eae, fsd/fsd_computation.py)
rather than the paper's prose wherever the two differ. Upstream is CC BY-NC-SA 4.0; this is a
non-commercial port with attribution.

    residuals   r_k = I - FRE_k(I)                       (fre.py; K constrained M x M filters)
    scales      r_k^(l) = bilinear downsample by 2^l,  l = 0 .. L-1
    model       r_k^(l)(x,y) ~ sum_{(m,n) in B x B \\ (0,0)} phi_k(m,n) r_k^(l)(x+m, y+n)
    fit         Phi = argmin sum_{x,y,l} (sum_k r_k^(l)(x,y) - sum_k rhat_k^(l)(x,y))^2
                subject to sum(phi_k) = 1 for every k

The fit is a linear least-squares problem with K linear equality constraints, so it has a
closed form (the KKT system below) -- the paper's "AdamW, 10,000 iterations" is the iterative
route to the same minimiser, and upstream's released code solves it directly too. Two things
the paper does not say and the code does: the target is the *sum* over k of the centre values
(one joint regression, not K independent ones -- that is what "optimize all filters jointly"
means in practice), and each phi_k carries a sum-to-one constraint.

Deliberate deviation: upstream resizes the residual maps so the short side is 1024 before
anything else. Here the input is already the 200x200 native-pixel crop the detector panel
reads (imports/preprocess_crop_cache.py, crop200_align16), and nothing is resized: on this
sample that resize would upsample the 200-px Artifact images 5x and shrink RAISE scans, which
makes resampling history a source cue. See DECISIONS.md D-006.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

__all__ = ["load_gray", "load_channels", "CHANNELS", "residuals", "downsample", "fit", "describe",
           "description_dim"]

RIDGE = 1e-5          # upstream's lambda_reg: Tikhonov term on X^T X, for numerical stability
CHUNK_ROWS = 8192     # rows of the design matrix materialised at once


def load_gray(path):
    """The crop as float64 luma in 0..255 -- PIL's "L" (ITU-R 601), as upstream converts."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


# What `describe` is run on. "luma" is upstream's (one PIL "L" plane). "rgb" describes R, G and B
# separately with the same luma-trained FRE and concatenates the three Phis, channel-major: a
# departure from the paper (DECISIONS.md D-011), there to see whether the colour planes carry
# generator or time information that luma averages away.
CHANNELS = {"luma": 1, "rgb": 3}


def load_channels(path, channels="luma"):
    """The planes `describe` is run on, each float64 in 0..255: [luma] or [R, G, B]."""
    if channels == "luma":
        return [load_gray(path)]
    if channels == "rgb":
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        return [rgb[..., c] for c in range(3)]
    raise ValueError(channels)


def residuals(gray, filters):
    """r_k = I - sum_{offsets} w_k * I over the valid region only.

    `filters` is (K, M, M) with a zero centre. Upstream convolves with zero padding and then
    crops M//2 from every side, which is exactly the valid region computed here -- no padded
    pixel ever reaches the residual. Returns (K, H-M+1, W-M+1).
    """
    K, M, _ = filters.shape
    windows = sliding_window_view(gray, (M, M))                 # (H', W', M, M)
    h, w = windows.shape[:2]
    pred = windows.reshape(h * w, M * M) @ filters.reshape(K, M * M).T
    centre = gray[M // 2: M // 2 + h, M // 2: M // 2 + w]
    return centre[None] - pred.T.reshape(K, h, w)


def downsample(x, factor):
    """torch F.interpolate(scale_factor=1/factor, mode="bilinear", align_corners=False,
    antialias=False) along the last two axes, for factor a power of two.

    With align_corners=False, output pixel i samples source coordinate factor*i + (factor-1)/2.
    For even factors that is exactly halfway between source pixels factor*i + factor/2 - 1 and
    factor*i + factor/2, so bilinear interpolation reduces to averaging those two per axis
    (four pixels in 2-D) -- no interpolation weights needed, and the result is bit-for-bit the
    same arithmetic. Output size is floor(n / factor), as torch's. Note that for factor >= 4 this
    *discards* most pixels (no antialiasing), which is what upstream does.
    """
    if factor == 1:
        return x
    assert factor > 1 and factor & (factor - 1) == 0, f"factor must be a power of two, got {factor}"
    lo, hi = factor // 2 - 1, factor // 2
    nh, nw = x.shape[-2] // factor, x.shape[-1] // factor
    rows = 0.5 * (x[..., lo::factor, :][..., :nh, :] + x[..., hi::factor, :][..., :nh, :])
    return 0.5 * (rows[..., lo::factor][..., :nw] + rows[..., hi::factor][..., :nw])


def description_dim(K, B):
    return K * (B * B - 1)


def _normal_equations(maps, B):
    """Accumulate X^T X, X^T y, y^T y and the row count over every scale's B x B patches.

    Row layout matches upstream: per patch, channel k major, then the B*B - 1 off-centre
    positions in row-major order. y is the sum over k of the patch centres.
    """
    K = maps[0].shape[0]
    n_feat = B * B - 1
    D = K * n_feat
    mask = np.ones(B * B, dtype=bool)
    mask[(B * B) // 2] = False

    XtX = np.zeros((D, D))
    Xty = np.zeros(D)
    yty = 0.0
    rows = 0
    for r in maps:
        padded = np.pad(r, ((0, 0), (B // 2, B // 2), (B // 2, B // 2)), mode="reflect")
        win = sliding_window_view(padded, (B, B), axis=(1, 2))   # (K, h, w, B, B)
        h, w = win.shape[1:3]
        win = win.reshape(K, h * w, B * B)
        for start in range(0, h * w, CHUNK_ROWS):
            chunk = win[:, start:start + CHUNK_ROWS]                 # (K, n, B*B)
            X = chunk[:, :, mask].transpose(1, 0, 2).reshape(-1, D)
            y = chunk[:, :, (B * B) // 2].sum(axis=0)
            XtX += X.T @ X
            Xty += X.T @ y
            yty += y @ y
            rows += len(y)
    return XtX, Xty, yty, rows


def fit(maps, B):
    """Phi from multi-scale residual maps [(K, h_l, w_l)]: the constrained joint least squares.

    KKT system for min ||y - X phi||^2 + ridge ||phi||^2 subject to sum(phi_k) = 1 per k.
    Returns (phi (K*(B*B-1),), mean squared model error without the ridge term, row count).
    """
    K = maps[0].shape[0]
    XtX, Xty, yty, rows = _normal_equations(maps, B)
    D = XtX.shape[0]
    n_feat = B * B - 1
    A = np.zeros((K, D))
    for k in range(K):
        A[k, k * n_feat:(k + 1) * n_feat] = 1.0
    lhs = np.block([[XtX + RIDGE * np.eye(D), A.T], [A, np.zeros((K, K))]])
    rhs = np.concatenate([Xty, np.ones(K)])
    phi = np.linalg.solve(lhs, rhs)[:D]
    sse = yty - 2 * phi @ Xty + phi @ XtX @ phi
    return phi, float(sse / rows), rows


def describe(gray, filters, B=11, L=3):
    """Phi for one grayscale image, flattened to K*(B*B-1), plus fit diagnostics.

    Diagnostics (all cheap by-products, kept for QC and as candidate covariates):
      residual_energy  mean r_k^2 at full scale, per k  (how much the FRE leaves unexplained)
      fit_mse          mean squared model error over all scales, without the ridge term
      phi_norm         ||Phi||_2  (ill-conditioned fits -- flat crops -- blow this up)
    """
    r = residuals(gray, filters)
    phi, mse, rows = fit([downsample(r, 2 ** l) for l in range(L)], B)
    diagnostics = {
        "residual_energy": (r ** 2).mean(axis=(1, 2)),
        "fit_mse": mse,
        "phi_norm": float(np.linalg.norm(phi)),
        "rows": rows,
    }
    return phi, diagnostics
