"""Checks the self-description pipeline is only as trustworthy as.  python -m selfdesc.selftest

  (a) residuals() equals an explicit zero-padded correlation cropped by M//2 -- upstream's
      conv2d(padding=M//2) followed by its border crop;
  (b) downsample() equals bilinear sampling at torch's align_corners=False source coordinates,
      written out independently, for factors 2 and 4 on odd sizes;
  (c) fit() equals a brute-force constrained least squares on an explicitly built design matrix
      (offset by offset, reflect padding, channel-major layout), solved through the constraint's
      null space rather than the KKT system;
  (d) fre.train with lambda = 0 lands on the closed-form constrained predictor, and every filter
      keeps a zero centre and unit sum;
  (e) the diversity term does its job: lambda = 0 collapses K filters onto one (minor singular
      values ~ 0), lambda > 0 does not.
"""

import numpy as np

from . import core, fre

__all__ = ["run"]


def _bilinear_reference(x, factor):
    """torch bilinear, align_corners=False, no antialias: src = (i + 0.5) * factor - 0.5."""
    def axis(a, n_in, ax):
        n_out = n_in // factor
        src = (np.arange(n_out) + 0.5) * factor - 0.5
        src = np.clip(src, 0, n_in - 1)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        t = src - i0
        a0, a1 = np.take(a, i0, axis=ax), np.take(a, i1, axis=ax)
        shape = [1] * a.ndim
        shape[ax] = n_out
        return a0 * (1 - t.reshape(shape)) + a1 * t.reshape(shape)
    return axis(axis(x, x.shape[-2], -2), x.shape[-1], -1)


def _brute_force_fit(maps, B):
    K = maps[0].shape[0]
    c = B // 2
    offsets = [(m, n) for m in range(-c, c + 1) for n in range(-c, c + 1) if (m, n) != (0, 0)]
    rows_X, rows_y = [], []
    for r in maps:
        padded = np.pad(r, ((0, 0), (c, c), (c, c)), mode="reflect")
        h, w = r.shape[1:]
        cols = [padded[k, c + m: c + m + h, c + n: c + n + w].ravel()
                for k in range(K) for (m, n) in offsets]
        rows_X.append(np.stack(cols, axis=1))
        rows_y.append(r.sum(axis=0).ravel())
    X, y = np.concatenate(rows_X), np.concatenate(rows_y)
    n_feat = len(offsets)
    # phi = phi0 + N z, phi0 feasible, N spanning {sum over each channel's block = 0}
    phi0 = np.full(K * n_feat, 1.0 / n_feat)
    A = np.kron(np.eye(K), np.ones((1, n_feat)))
    N = np.linalg.svd(A)[2][K:].T
    # match fit()'s ridge so the two minimise the same objective
    Xr = np.vstack([X, np.sqrt(core.RIDGE) * np.eye(K * n_feat)])
    yr = np.concatenate([y, np.zeros(K * n_feat)])
    z = np.linalg.lstsq(Xr @ N, yr - Xr @ phi0, rcond=None)[0]
    return phi0 + N @ z


def run(seed=0):
    rng = np.random.default_rng(seed)
    lines = []

    # (a)
    K, M = 3, 5
    gray = rng.uniform(0, 255, (23, 19))
    filters = rng.normal(size=(K, M, M))
    filters[:, M // 2, M // 2] = 0
    filters /= filters.reshape(K, -1).sum(axis=1)[:, None, None]
    padded = np.pad(gray, M // 2)
    ref = np.empty((K,) + gray.shape)
    for k in range(K):
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                ref[k, i, j] = gray[i, j] - (padded[i:i + M, j:j + M] * filters[k]).sum()
    ref = ref[:, M // 2:-(M // 2), M // 2:-(M // 2)]
    err = np.abs(core.residuals(gray, filters) - ref).max()
    assert err < 1e-9, err
    lines.append(f"(a) residuals == padded conv + crop     max |diff| = {err:.1e}")

    # (b)
    x = rng.normal(size=(2, 37, 29))
    errs = [np.abs(core.downsample(x, f) - _bilinear_reference(x, f)).max() for f in (2, 4)]
    assert max(errs) < 1e-12, errs
    lines.append(f"(b) downsample == torch bilinear       max |diff| = {max(errs):.1e} (x2, x4)")

    # (c)
    maps = [rng.normal(size=(2, 17, 15)), rng.normal(size=(2, 8, 7))]
    phi, _, _ = core.fit(maps, 3)
    ref = _brute_force_fit(maps, 3)
    err = np.abs(phi - ref).max()
    assert err < 1e-8, err
    sums = phi.reshape(2, -1).sum(axis=1)
    assert np.allclose(sums, 1)
    lines.append(f"(c) KKT fit == null-space lstsq        max |diff| = {err:.1e}")

    # (d)
    M = 5
    img = np.cumsum(np.cumsum(rng.normal(size=(64, 64)), 0), 1)       # smooth, image-like
    win = np.lib.stride_tricks.sliding_window_view(img, (M, M)).reshape(-1, M * M)
    R = win.T @ win / len(win)
    F0, info0 = fre.train(R, M, 3, lam=0.0)
    r_cc, r_nc, R_nn = fre._split(R, M)
    n = M * M - 1
    kkt = np.linalg.solve(np.block([[2 * R_nn, np.ones((n, 1))], [np.ones((1, n)), 0]]),
                          np.r_[2 * r_nc, 1.0])[:n]
    V0 = np.delete(F0.reshape(3, -1), (M * M) // 2, axis=1)
    err = np.abs(V0 - kkt).max() / np.abs(kkt).max()
    assert err < 1e-6, err
    assert np.all(F0[:, M // 2, M // 2] == 0) and np.allclose(F0.reshape(3, -1).sum(1), 1)
    lines.append(f"(d) FRE(lambda=0) == closed form      rel |diff| = {err:.1e}")

    # (e)
    F1, info1 = fre.train(R, M, 3, lam=1.0)
    minor0 = max(info0["singular_values"][1:])
    minor1 = min(info1["singular_values"][1:])
    assert minor0 < 1e-4 and minor1 > 100 * minor0, (minor0, minor1)
    lines.append(f"(e) diversity: minor sv {minor0:.1e} (lambda=0) -> {minor1:.3f} (lambda=1)")

    print("\n".join(lines))
    print("selfdesc selftest: all checks passed")
    return lines


if __name__ == "__main__":
    run()
