# DECISIONS.md

Running log of decisions and results for the thesis prototype. Newest entries go at the **top** of
each section. `CLAUDE.md` holds the standing project brief; this file holds what we *decided and
found* as the work progresses.

## How to use this file (for Claude)

- **Read it at the start of a task**, before proposing an approach — don't re-derive or relitigate
  anything already logged here.
- **Append an entry when** a decision is made (method, parameter, scope, tooling, data handling) or
  an experiment produces a result worth keeping. Do it in the same change as the work, not later.
  If unsure whether something qualifies, ask Brian.
- **Never rewrite history.** If a decision is reversed, add a new entry with status `Supersedes D-nnn`
  and mark the old one `Superseded by D-nnn`. Results that turn out to be wrong get the same treatment.
- **Tag claims** with the evidence convention from CLAUDE.md: [S] sourced, [I] inference,
  [R] recalled, [?] unknown. Results must say which experiment directory / commit produced them.
- Keep entries short and concrete: numbers, paths, commit hashes. No hedging filler.
- Do not duplicate CLAUDE.md. Settled core idea and scope cuts live there; only log changes to them.

## Entry templates

```
### D-nnn · YYYY-MM-DD · <short title>
- **Status:** Accepted | Superseded by D-nnn | Open
- **Decision:** what we chose.
- **Why:** the reasoning / alternatives rejected.
- **Consequences:** what this commits us to, what it rules out, follow-ups.
- **Refs:** commits, files, papers.
```

```
### R-nnn · YYYY-MM-DD · <short title>
- **Experiment:** directory / notebook / commit that produced it.
- **Setup:** data, panel, sample size, key parameters.
- **Result:** the numbers (Cllr, FPR, T* error, ...), with a one-line plain reading.
- **Caveats:** confounds, sensitivity, what it does *not* show.
- **Next:** what it implies.
```

Numbering is continuous and never reused. Next free IDs: **D-008**, **R-002**.

---

## Decisions

### D-007 · 2026-10-10 · Granularity = PCA components kept after family centring
- **Status:** Accepted
- **Decision:** For the self-description experiment, within-family structure is read off a
  *family-centred* view: subtract each family's mean Φ from its own images, fit **one** PCA on the
  pooled remainder of the **fake** families only, and keep the first *g* components. *g* is the
  granularity setting, swept over {1, 2, 3, 5, 10, 20, 50}. Authentic images are centred on their own
  mean and projected onto these axes, never used to fit them. Within-family generator separability
  is balanced leave-one-out 5-NN accuracy with a permutation null (labels shuffled within the
  family, neighbour graph fixed), plus KMeans ARI/NMI and Calinski–Harabasz. Because generators
  share origin datasets unevenly, the headline within-family test is **stratified**: only
  generators sharing family *and* origin dataset (≥5 images each) are compared, neighbours are
  searched inside each cell, and labels are shuffled inside each cell
  (`analysis.stratified_separability`).
- **Why:** [S: Brian's spec] "subtract each family's mean description from its images, run PCA on
  the remainder, and use the number of components kept as the granularity setting". Balanced
  accuracy because authentic is half the sample; a plain-accuracy majority baseline of 50% hides
  family structure. A pooled PCA (rather than one per family) gives every family the same
  axes, so per-family results are comparable. Fitting on fakes only keeps the authentic sources'
  processing differences out of the axes.
- **Consequences:** release date is shown as colour/order only; relating components to date is
  a later notebook. If families barely differ in mean Φ, the centred and raw PCA bases nearly
  coincide (see R-001).
- **Refs:** `selfdesc/analysis.py`, `self_descriptions.ipynb`.

### D-006 · 2026-10-10 · Forensic self-descriptions as an image representation (`selfdesc/`)
- **Status:** Accepted
- **Decision:** Represent each image by its forensic self-description Φ (Nguyen, Azizpour &
  Stamm, CVPR 2025, arXiv 2503.21003), ported to numpy from the authors' code
  (`ductai199x/Forensic-Self-Descriptions-CVPR25` @ 50f2eae). Per-image Φ is the closed-form
  constrained least-squares (KKT) solve upstream uses: joint target Σ_k centre values, sum-to-one
  per φ_k, ridge 1e-5, fp64. Choices:
  1. **Input = the panel's `crop200_align16` crops, no resize.** Upstream resizes residuals to a
     1024 short side; on this sample that would upsample 200-px images 5× and shrink RAISE,
     making resampling history a source cue. [Brian, 2026-10-10]
  2. **One FRE trained per K** on authentic `split == "train"` crops (self-supervised, real images
     only; no synthetic image or label), frozen and hashed, tracked in git
     (`experiments/selfdesc-sample/fre/`). Not a detector, so outside the "no training new
     detectors" cut. [Brian, 2026-10-10]
  3. **λ calibrated, not copied.** [S: paper §5.1] gives λ = 1 for a loss written as a sum over
     pixels, normalisation unstated. [I] With per-pixel-mean energy, λ = 1 leaves the filters
     nearly identical (minor singular values ≈ 0.13), while the released K=8 FRE has ≈ 0.95 [S:
     decoded v1.2.0 `fre.pt`, sha256 d95b9c50…]. λ is bisected so our K=8, M=15 FRE matches the
     released mean minor singular value, giving **λ = 52.16**. It is held fixed for all K, M.
     α = 1e-3.
  4. M = 11 as the paper states (release uses 15). The released FRE is kept as one anchor config
     (`upstream_K8_M15_B11_L3`), fetched on demand and not redistributed (CC BY-NC-SA 4.0).
  5. No upstream MLP feature transforms; Φ is the raw parameter set.
  6. Analysis scales features by median/IQR by default (Φ has heavy-tailed coordinates from
     near-collinear residual channels); standard and unscaled are re-run as sensitivity checks.
- **Grid:** K ∈ {1,2,4,8,16}, B ∈ {5,7,11}, L ∈ {1,2,3}, one at a time around (8, 11, 11, 3), plus
  the upstream anchor.
- **Consequences:** extraction (`python -m selfdesc.sweep`) is the expensive, resumable step;
  `self_descriptions.ipynb` only loads. Throughput is ~1.2 img/s at K=8 on the M5. The server run
  reuses the local FREs via `--fre-dir`. Every image's metadata (generator, family, date,
  origin dataset, native size, source and stored format) lives in `images.parquet`.
- **Refs:** `selfdesc/`, `selfdesc/README.md`; `python -m selfdesc.selftest` (5 checks against
  independent references, all pass).

### D-005 · 2026-10-09 · Three branches with separate jobs
- **Status:** Accepted
- **Decision:** `main` = all experiments; `downloader` = full-dataset (360k) build onto server
  `/data`; `server-run` = `runner/` unattended scoring pipeline + health dashboard. None forks another.
  Old `stratified-run` split into main/server-run; tip kept as tag `backup/stratified-run-2026-10-09`.
- **Why:** `stratified-run` mixed experiment code with runner infra. [S: CLAUDE.md]
- **Consequences:** new work goes on the branch whose job it is. `server-run` drives a `main`
  checkout's `detectors/` via `PANEL_REPO`.
- **Refs:** b6ee0fe.

### D-004 · 2026-09-25 · Panel and scoring run on the server GPUs; local work uses the in-repo sample
- **Status:** Accepted
- **Decision:** Detector scoring and training happen on the lab server (2× RTX 3080), dataset on
  `/data`. Local day-to-day work uses the in-repo 612+612 sample; `AIGENBENCH_DATA_ROOT` switches
  `DATA_DIR` to the full set.
- **Why:** published detector code assumes CUDA; MPS op coverage is rough; network-mounted random
  reads would bottleneck. [S: CLAUDE.md]
- **Consequences:** panel installer is driven by `panel.json`; score sidecars record `batch_size`
  and dtype so server runs are reproducible.
- **Refs:** 65e3bc4, 989fb30, cea132d, e217742.

### D-003 · 2026-09-25 · AEROBLADE scored with two autoencoders, asserted
- **Status:** Accepted
- **Decision:** AEROBLADE is scored with two autoencoders and the code asserts it; fp16 available on
  CUDA to match upstream.
- **Refs:** 57eb51d, cea132d, 0deb3b3 (bounded joblib cache, stand down on low disk).

### D-002 · 2026-09-26 · Full-set scoring plus a 36k stratified draw
- **Status:** Accepted
- **Decision:** Score the full AI-GenBench set (355,638 images) and a 36k stratified draw
  (`experiments/stratified-36k`), manifests carry `origin_dataset`/`split`.
- **Why:** the stratified draw gives a cheaper iteration loop; the full set is the headline run.
- **Consequences:** experiment manifests and full-scale score CSVs are gitignored; only sidecars
  (`manifest.meta.json`, score metadata) are tracked.
- **Refs:** 89d93c6, 18c4979, 3c8cb73.

### D-001 · 2026-08-31 · Core idea, architecture A, scope cuts
- **Status:** Accepted
- **Decision:** As stated in CLAUDE.md: LR(F,T) over G≤T, architecture A primary (frozen detector
  panel + per-family score models + max aggregation), B as fallback; images only, no new detector
  training, no live retrieval.
- **Why / consequences:** see CLAUDE.md "Core idea" and "Scope cuts". Do not relitigate.
- **Refs:** CLAUDE.md; 47b9b32 (Architecture A).

---

## Results

### R-001 · 2026-10-10 · Forensic self-descriptions on the local sample: weak family structure, robust within-family generator structure
- **Experiment:** `self_descriptions.ipynb` on `experiments/selfdesc-sample/` (manifest sha256
  4b7dca1c…; FREs + `lambda.json` in `experiments/selfdesc-sample/fre/`); code `selfdesc/` (D-006, D-007).
- **Setup:** in-repo sample, 612 authentic + 612 fake (17 per generator, 36 generators), crop200_align16
  grayscale crops, all JPEG-q95 re-encoded. 10 configs: K ∈ {1,2,4,8,16}, B ∈ {5,7}, L ∈ {1,2}
  around (K8, M11, B11, L3), plus the released FRE (K8, M15). Robust scaling, with standard and none as
  checks (12 settings). Balanced LOO 5-NN with permutation nulls (500 for the default, 200 in the sweep).
- **Result (default K8_M11_B11_L3 unless stated):**
  1. Family structure weak. Between-family share of fake-row variance **2.2 %**. Family 5-NN among
     fakes 0.30 (chance 0.17, null q95 0.19), KMeans NMI 0.05. Family excess over null is +0.04…+0.13
     across all 12 settings. The paper's tight per-source t-SNE clusters do not reproduce here.
  2. Φ tracks image source as much as generator. Authentic sources separate at 0.70 (chance 0.33);
     among fakes, origin dataset 0.165 (chance 0.062) and native-size bucket 0.49 (0.33) separate about
     as well as generator 0.116 (0.028).
  3. Raw and family-centred views nearly coincide (little family mean to remove), so centring adds
     nothing at this sample's family separation.
  4. **Holds in every setting:** generators sharing family *and* origin dataset (12 cells, 484 images;
     source format also constant in each) still separate. Default 0.38 vs null 0.19 at g ≥ 10
     (p = 0.002). Excess over null is +0.09…+0.20 at g = 5 and +0.11…+0.20 at g = 20 in all 12 settings,
     each at the minimum attainable p. Clearest: inpainting/Artifact 0.59 (chance 0.33), GAN/Artifact
     0.21 (0.08), FLUX Dev vs Schnell 0.62 (0.50).
  5. Granularity: within-family separability rises up to g ≈ 10 and plateaus near the full-Φ ceiling.
  6. **Does not hold across settings:** per-family detail. Spearman of the per-cell pattern with the
     default is −0.02 (K16) … 0.76 (L2), against 0.94–0.95 for the alternative scalings. L = 1 is weakest,
     in line with [S: paper Tab. 5, "No multi-scale"]. K16 is no better than K8 (median ‖Φ‖ 382 vs 129).
- **Plain reading:** the precondition for the time hypothesis holds weakly but robustly: Φ varies
  between same-family generators with dataset and format fixed. Whether that variation is
  time-specific is untested (deferred by design).
- **Caveats:** n = 17 per generator; uniform JPEG-q95 re-encode (the paper's JPEG test [S: §7, Tab. 6]
  covers detection AUC only, not attribution); 200 × 200 crops; FREs trained on 306 authentic crops.
  [I] The weak family structure may be specific to this pipeline; untested.
- **Next:** run on the server's stratified 36k with the same frozen FREs (`--fre-dir
  experiments/selfdesc-sample/fre`); then the release-date notebook, restricted to within-origin
  comparisons.


---

## Open questions

Tracked in detail in CLAUDE.md ("The central open question", "Other open threads"). Move an item
here only when it gets a concrete next step, and link the D-/R- entry that resolves it.
