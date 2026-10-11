# `selfdesc/` — forensic self-descriptions

Each image is represented by its **forensic self-description** Φ = {φ₁ … φ_K}
(Nguyen, Azizpour & Stamm, CVPR 2025, arXiv 2503.21003): the linear filters that best model
the image's *own* forensic residuals. No generator label is used anywhere in producing Φ. It
is a per-image description, not a discriminative embedding. Analysis lives in
`self_descriptions.ipynb`, which only loads what this package writes.

| module | what it holds |
|---|---|
| `fre.py` | the residual extractor: K constrained M×M predictors learned on **real images only**; λ calibration; decoder for the authors' released weights |
| `core.py` | Φ for one image: residuals → L scales → B×B joint constrained least squares (closed-form KKT) |
| `extract.py` | CLI: Φ for every manifest row under one config → `descriptions/<config>/`; resumable shards |
| `sweep.py` | CLI: calibrate λ, train the FREs a grid needs, extract every config |
| `analysis.py` | notebook helpers: loading with alignment checks, scaling, family centring, PCA, separability + permutation nulls |
| `selftest.py` | `python -m selfdesc.selftest`: each piece checked against an independent reference |

## Run

```sh
python imports/build_manifest.py --out experiments/selfdesc-sample/manifest.csv
python -m selfdesc.sweep --manifest experiments/selfdesc-sample/manifest.csv \
    --out experiments/selfdesc-sample --workers 10
```

On the server, reuse the FREs frozen here so the two runs differ only in images:

```sh
python -m selfdesc.sweep --manifest experiments/stratified-36k/manifest.csv \
    --out experiments/selfdesc-36k --fre-dir experiments/selfdesc-sample/fre --workers 40
```

Throughput (fp64, one BLAS thread per worker): ~1.2 images/s at K=8, B=11, L=3 across the M5's
10 cores; cost grows with (K·(B²−1))², so K=16 is ~4× slower. CPU, not GPU: the 3080s are slow
at fp64, and upstream insists on fp64 because the solve amplifies float32/TF32 drift.

## Grid

One thing at a time around the paper's setting (K=8, M=11, B=11, L=3): K ∈ {1,2,4,8,16},
B ∈ {5,7,11}, L ∈ {1,2,3}, plus the authors' released FRE (K=8, M=15) as an external anchor.
Config ids read `K8_M11_B11_L3`, `upstream_K8_M15_B11_L3`.

## Output (`experiments/<name>/`)

```
images.parquet                         one row per image, shared by every config: generator, family,
                                       release_date, origin_dataset, split, native size, source_format
                                       (+ its basis), stored_format, crop box, crop_policy
fre/fre_K{K}_M{M}.npz + .meta.json     frozen FREs (tracked in git) + how they were trained; lambda.json
descriptions/<config>/descriptions.npz X (n, K·(B²−1)) float32, image_id
descriptions/<config>/diagnostics.parquet  per-image ‖Φ‖, model error, residual energy per k
descriptions/<config>/meta.json        config, FRE + manifest hashes, crop policy, commit, timing
```

## Where this departs from upstream, and why (DECISIONS.md D-006)

* **No resize.** Upstream resizes every residual map so the short side is 1024. The input here is
  the panel's native-pixel 200×200 crop (`crop200_align16`). On this sample, the resize would
  upsample 200-px images 5× and shrink RAISE scans, which makes resampling history a source cue.
* **Our own FRE per K.** The released weights fix K=8, so sweeping K means training an FRE per
  K. Training is self-supervised on authentic `split == "train"` crops only; no synthetic image or
  label is involved.
* **λ is calibrated, not copied.** The paper's λ=1 is stated against a loss whose normalisation is
  not given. With a per-pixel-mean energy, λ=1 leaves the K filters nearly identical. λ is bisected
  so that our K=8, M=15 FRE matches the released FRE's spectral diversity (mean minor singular
  value 0.951), giving λ ≈ 52.2. It is then held fixed for all K and M.
* **M = 11**, as the paper states. The released weights use 15; the upstream anchor config keeps
  them as they are.
* **No learned feature transforms.** Upstream's release adds MLP projections before its GMM. Φ
  here is the raw parameter set the paper defines.

Upstream code is CC BY-NC-SA 4.0. This is a non-commercial port with attribution. The released
weights are not redistributed: `fre.fetch_upstream` downloads them and checks their hash.
