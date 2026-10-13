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

Numbering is continuous and never reused. Next free IDs: **D-012**, **R-003**.

---

## Decisions

### D-011 · 2026-10-13 · Five diagnostics for the within-family time null, and what the server run covers
- **Status:** Accepted. Logged before any server result. Amends D-010's config list (only K8 runs).
- **Decision:** Alongside D-008/D-009's pre-registered confirmatory test, the server run tests five
  explanations for R-002's null. Each one has a test, and its reading is fixed in advance. All of
  them are exploratory (uncorrected, reported in full). Only D-008's four primary specs carry Holm.
  1. **Lineage, not date** (`selfdesc/lineage.csv`). Each generator gets an output-stage label
     (`stage`: GAN transposed-conv / upsample-conv / alias-free / no-upsampling; diffusion
     pixel / latent / unknown) and a `codebase` label. Most are [R]. Latent Diffusion's KL-f8 is [S]
     (CompVis `txt2img-1p4B-eval.yaml`: `AutoencoderKL`, `embed_dim: 4`). Midjourney and DALL-E 3
     are `unknown` and excluded. **L1:** T1's machinery with lineage classes in place of time
     bins (classes spanning ≥ 2 release groups). Null: labels permuted among generators,
     conditional on every class still spanning 2 groups. **L2:** a rank regression of T2's
     held-out pair divergence on [different lineage, |Δ date|], each term with its own
     permutation null. *Supports lineage:* L1 `stage` p ≤ 0.05 and L2 b_lineage p ≤ 0.05 with
     b_date p > 0.05. *Against:* L1 at its null.
  2. **Power** (T5). The drift of release-group means over the date range, in within-generator
     SDs (Mahalanobis), tested against a date-permutation null. It is bounded above by a group
     bootstrap (95th percentile). It comes with a simulation, from the observed scatter, of how
     many release groups a drift of 0.5/1/2 SD needs. *A bound < 1 SD* means moderate drift is
     ruled out. *A bound ≥ 1 SD* means the question is power-limited. AI-GenBench cannot add
     groups (36 generators). Only new generators can (the post-2024 fill).
  3. **Input pipeline** (arms): 200 fakes per generator (7,200 rows, seeded), K8, in a 2 × 2 design.
     The two factors are **q95 crops** (the panel's) vs **native crops**, and **luma** vs
     **RGB**. Native crops have the same box and alignment, cut from the published AI-GenBench
     parquet source bytes with `prepare_image`'s decode and alpha composite but no JPEG re-encode
     (`imports/native_crop_cache.py`; QC = PSNR vs the q95 crop, refused below 30 dB). RGB means
     Φ for R, G and B separately with the frozen luma FRE, concatenated: a departure from the
     paper, which uses luma only. Every ArtiFact image (GAN-heavy) already arrives as a 200-px
     JPEG, so no pipeline of ours can restore it. Every arm is therefore also read without
     ArtiFact. Readout per arm: S1 (family 5-NN among fakes; generator 5-NN inside family ×
     origin cells), T1 primary, T5. *Pipeline was hiding signal:* native arms raise S1 and T1/T5.
     *Pipeline not the reason:* S1 rises, T1/T5 do not. *No effect:* neither rises.
  4. **Centring.** Single-origin T1 specs, where no centring choice can matter. These are
     GAN∩ArtiFact (already in the D-008 grid), GAN∩Forensynths and diffusion∩GenImage at 3 bins,
     and diffusion∩ArtiFact at 2 bins (4 groups). Origin is matched on the source name, ignoring
     split suffixes. They are read together with the family-only centring spec. *Origin:* a
     family-only signal that is absent within single origins.
     **Found while preparing (local sample, K8, T5, 500 date permutations):** under family ×
     origin centring the observed diffusion drift sits *below* its null (Φ 0.46 SD vs null mean
     1.03, p 0.99; φ̄ 0.29 vs 0.96, p 1.00). Under family-only centring it is above (Φ 2.97 vs
     1.70, p 0.012; φ̄ 2.80 vs 1.57, p 0.008). GAN is null both ways. [I] Origin datasets are
     tied to release dates (ArtiFact early, SFHQ-T2I = FLUX only), so centring on them
     removes date-aligned variation. The date-permutation null is then not exchangeable and is
     conservative. This plausibly applies to D-008's primary T1 too, which uses the same
     centring, so its confirmatory p-values lean toward the null. Family-only centring has the
     opposite problem: origin can pass as time. Only single-origin tests escape both, so T5 also
     runs under family-only centring and within every single origin and size band. The
     confirmatory spec is left as registered; this caveat goes with its result.
  5. **Resolution** (diffusion). T1 at 2 and 3 bins, T5, and a size-only T1 inside native
     short-side bands ≤ 256 px (DDPM, ADM, Palette, VQ-Diff, DDGAN/Glide/LDM, IF: 6 groups) and
     ≥ 512 px (MJ, SD1.4/1.5/2.1, SDXL, DALL-E 3, FLUX, part of IF). *Not just resolution:* the
     φ̄ signal survives in a band where size-only T1 is near its null. The family-only diffusion
     drift above is the one to check this way: FLUX, DALL-E 3 and SDXL are both the latest and
     the largest outputs.
- **Scale ("large enough"):** the unit is the release group, so per-generator n beyond a few
  hundred adds little. The 36k draw (500 per generator) for the confirmatory run and diagnostics
  1, 2, 4 and 5; 200 per generator for the arms. **Only K8 is extracted.** The rest of D-006's
  config grid is a robustness check that none of the five questions needs, so it is deferred
  (user, 2026-10-13). It would have been the longest run (~3–4 h without K16; K16 alone was
  measured on the server at 9.3 s/image/core vs 2.8 for K8, one core, nice 10).
- **Order** (`experiments/selfdesc-d011.queue`, `./runner/run.sh selfdesc queue`): 36k K8 +
  full suite; q95 arms; native arms. Estimates [I] from the measured 2.8 s/image/core and about
  24 physical cores (48 threads):
  - 36k K8 about 1–1.5 h (D-010 said ~40 min: it counted 40 threads as cores), full temporal
    suite 1–1.5 h;
  - the arms about 1.5 h with their suites.
  - Total about 4–5 h. Native crops are built beforehand: a one-off read of the 72 shards (~35 GB).
  - **Corrected 2026-10-13, after the server smoke run:** K8 throughput *falls* with more
    concurrent processes (one-core 0.45 img/s; 12 processes 3.05; 24 2.10; 36 1.68). Each fit
    streams hundreds of MB of temporaries, so the processes contend for cache and memory, not
    cores. Extraction therefore runs on 12 workers (server-run `SELFDESC_WORKERS`), and the
    temporal stage, whose GMM fits are cache-sized, runs on 24. At ~3 img/s:
    - 36k K8 about 3.3 h;
    - each arm about 2.7 h (RGB = 3× luma);
    - temporal about 1.5–2 h in total.
    The total is **~10–11 h**, not 4–5. The native crops took one pass (7,200 rows, PSNR vs q95
    ≥ 40 dB, median 47–54).
- **Why:** user request (2026-10-13) to test the five explanations for the null in R-002, plus
  the pending confirmatory run, at a size that answers them rather than the full 355k.
- **Consequences:**
  - `extract.py`/`sweep.py` gain `--channels`.
  - `temporal.py` gains `--suite {full, arms, quick}`, subsets (`origin=`, `origin!=`, `short<=`,
    `short>=`), L1, L2, T5, S1, and per-config output dirs (`temporal_<config>`).
  - An untestable subset returns NaN with a reason instead of crashing.
  - The runner gains `--channels/--temporal-configs/--suite` and a queue (server-run
    `runner/selfdesc_queue.py`).
  - R-002's local numbers stay valid; its code paths for the D-008 statistics are unchanged.
    Lineage labels must not be edited after results are seen without a new entry (the cache
    key includes the CSV hash).
- **Refs:** `selfdesc/lineage.csv`, `selfdesc/temporal.py`, `imports/native_crop_cache.py`,
  `experiments/selfdesc-d011.queue`; server-run `runner/selfdesc.py`, `runner/selfdesc_queue.py`.

### D-010 · 2026-10-11 · Server run of self-descriptions + temporal tests through the runner
- **Status:** Accepted; config list amended by D-011 (K8 only; rest of the grid deferred)
- **Decision:** Run the self-description extraction (all 10 configs) and the temporal tests on
  **stratified-36k** (500 images/generator), on the lab server, through the `server-run` branch's
  runner. It is a new CPU job (`runner/selfdesc.py`, `./runner/run.sh selfdesc ...`) beside the
  GPU panel job. Settings:
  - at most **40 of 48 cores**, never the last 4, lowered live as other users' load rises
    (agreed with Brian); every subprocess runs under `nice 10`;
  - the FREs frozen on the local sample are reused, and `--frozen` makes training a new one an
    error, so the server and local runs differ only in their images;
  - `K8_M11_B11_L3` is extracted first, since the confirmatory test (D-008/D-009) needs only it;
  - temporal: the pre-registered primary spec first, 1000 release-group permutations, every block
    cached for resumability.
- **Why:** user request (2026-10-11). The full 355k would take ~40 h of extraction for all
  configs and more than the confirmatory test needs.
- **Consequences:** main gains `extract.py --shard/--assemble/--shard-size`, `sweep.py
  --frozen/--plan-json` and the temporal cache. Output lands in
  `experiments/selfdesc-36k/` on the server's main checkout. Estimate [I]: ~2–3 s/image/core for
  K8 on the 2014 Xeons, so ~40 min for K8 and ~5 h for all configs, then ~2–3 h of temporal.
  Untested on the server until the smoke run.
- **Refs:** server-run `runner/selfdesc.py`, `runner/run.sh`; main `selfdesc/extract.py`,
  `selfdesc/sweep.py`, `selfdesc/temporal.py`.

### D-009 · 2026-10-11 · Release groups, not generators, are the unit in the time tests
- **Status:** Accepted. Supersedes D-008's holdout and permutation unit; the rest of D-008 stands.
- **Decision:** Generators released within **7 days** of each other (single-linkage on date) form
  one *release group*. T1 holds out a whole group, bins never split a group (members share the
  group's earliest date), T1 is summarised over groups, and the permutation null shuffles dates
  **among groups**. T2 drops within-group pairs and permutes at group level. Locally: GAN 13
  generators → 12 groups (the two Diffusion-GAN variants merge); diffusion 16 → 13 (FLUX Dev +
  Schnell; DDGAN + Glide + Latent Diffusion, 2021-12-16..21). The pre-registered server spec
  (D-008) uses this unit.
- **Why:** found in the first smoke test, before any primary run. FLUX Dev and FLUX Schnell (one
  release, one day apart, same source dataset) gave the two largest log-LRs. Holding one out left
  its near-twin in the training bin, while under the generator-level null the two almost never
  share a bin, so the null under-represents that advantage and p is inflated. The D-008 pilot's
  diffusion φ̄ result (bin accuracy 0.62, p≈0.02–0.05) carries this bias. With release groups the
  same smoke test gives log-LR +0.29 vs null −0.04, p≈0.2 (40 permutations, indicative only).
  [I] The 7-day window also merges unrelated same-week releases (DDGAN, Glide, LDM). That is
  conservative: it costs units but cannot create a spurious effect.
- **Consequences:** effective n drops to the number of release groups (12 GAN, 13 diffusion).
- **Amended 2026-10-11, after the four primary specs had run once:** time bins hold an equal
  number of **release groups**, not generators. The first full run crashed when a permuted 4-bin
  split left one group alone in a bin. Under the generator-count rule, a 3-generator group can
  starve a bin, which is inconsistent with groups being the unit. On the real dates, the primary
  3-bin partition is identical for diffusion and moves one boundary generator for GAN. The
  first-run primary numbers (generator-count bins) are recorded in R-002 next to the final ones.
- **Refs:** `selfdesc/temporal.py` (`release_groups`, `RELEASE_WINDOW_DAYS`, `time_bins`).

### D-008 · 2026-10-11 · Testing for time structure within a family: GMM per time bin, generator as the unit
- **Status:** Accepted; holdout/permutation unit superseded by D-009 (release groups)
- **Decision:** Whether the distribution of self-descriptions shifts over time within a family is
  tested in `self_descriptions_time.ipynb` via `selfdesc/temporal.py`:
  - **Unit of evidence = generator.** Time is a property of generators, so a temporal claim must
    transfer to an *unseen* generator. Image-level CV would only re-find generator identity
    (R-001). All p-values come from **generator-level permutations of release dates** within the
    family (re-bin, rerun). Effective n = number of generators (GAN 13, diffusion 16 locally).
  - **T1 (headline):** equal-generator-count time bins (ties kept together). Leave one generator
    out, fit one GMM per bin on the rest (diag covariance, components by BIC ≤ 4, reg_covar 1e-3), score
    the held-out generator under each bin. Statistic: mean over generators of the log-likelihood
    ratio of the true bin vs the mean of the others (nats/image). Bin accuracy and mean |bin error|
    are also reported.
  - **T2:** pairwise held-out divergence between (generator × origin) cells vs |Δ release date|,
    Spearman, same permutation null. Reported for all pairs and for same-origin pairs only.
  - **T3/T4 (descriptive):** what each bin's GMM components contain; BIC component count per
    generator.
  - **Features:** full Φ and φ̄ (mean over k of φ_k). The residual channels are ~0.93 correlated,
    so φ̄ is the well-conditioned part and φ_k − φ̄ holds the cancelling coefficients. Robust
    scaling, then centring on **family × origin dataset** means (removes origin-level location
    shifts), then pooled PCA on fakes, g = 10.
  - **Primary (local):** {diffusion, GAN} × {Φ, φ̄}, 3 bins, g = 10, 1000 permutations,
    Holm-corrected over the 4. **φ̄ was promoted after a read-only pilot on this same sample**
    (diffusion φ̄ 0.62 bin accuracy vs null ≈0.27; GAN at null), so local p-values for φ̄ are
    optimistic and the local run is exploratory.
  - **Pre-registered confirmatory spec for the server run (stratified-36k, same frozen FREs):**
    exactly the primary spec above (both families, both feature sets, family×origin centring,
    3 bins, g = 10, T1 statistic = mean log-LR, 1000 generator-level permutations, Holm over 4),
    plus T2 on same-origin pairs. Logged before that run; any deviation gets its own entry.
- **Why:** user request (2026-10-11): GMM per time bin, components as clusters. Quality as an
  alternative explanation is out of scope for now (user).
- **Consequences:** inpainting (3 generators) and other (2) cannot be binned; descriptive only.
  Sensitivity grid (200 permutations): bins 2/4, g 5/20, family-only centring, GAN∩Artifact, Φ blocks
  (rings, offset parity, deviations), the 10 configs.
- **Refs:** `selfdesc/temporal.py`, `self_descriptions_time.ipynb`.

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

### R-002 · 2026-10-11 · Time structure within families: not demonstrated; GAN rules out large drift, diffusion weak and resolution-confounded
- **Experiment:** `self_descriptions_time.ipynb` on `experiments/selfdesc-sample/temporal/`
  (`python -m selfdesc.temporal --exp experiments/selfdesc-sample --perms 1000`, 1530 s; size check
  via `--only size`, 515 s). K8_M11_B11_L3, design per D-008 as amended by D-009.
- **Setup:** 272 diffusion images (16 generators → 13 release groups), 221 GAN (13 → 12). φ̄ and
  full Φ; family×origin centring; g = 10; 3 equal-group time bins; diag GMMs with BIC ≤ 4; release
  groups held out and permuted. 1000 permutations for primary, 200 grid, 100 other configs.
- **Result:**
  1. Primary (Holm over 4): diffusion φ̄ log-LR +0.29 vs null −0.14, p 0.096 (Holm 0.38), bin acc
     0.46 (chance 0.33). Diffusion Φ −0.02 vs −0.12, p 0.28. GAN Φ −0.13 vs −0.12, p 0.50. GAN φ̄
     −0.22 vs −0.18, p 0.53. **None significant.** First, crashed run with generator-count bins
     (D-009 amendment): diffusion Φ p 0.237, φ̄ p 0.118; GAN Φ p 0.659, φ̄ p 0.606. Same conclusion.
  2. Power (planted drift on real features, 5 directions × 99 perms): a drift of 2 within-generator
     SDs over the date range is detected with power 0.8 (GAN) to 1.0 (diffusion); ≤ 1 SD gives
     0–0.4.
  3. Diffusion φ̄ excess is positive in 12/14 sensitivity settings, p ≤ 0.05 in 2 (K2; upstream
     FRE), uncorrected, and gone with 2 bins and with B = 5. All settings reuse one sample, so this is
     robustness to analysis choices, not replication.
  4. **Resolution confound (diffusion):** generator native short side tracks release date
     (Spearman 0.79; GAN 0.22). Log native size alone predicts the bin (acc 0.69, p 0.018). With
     size regressed out (linear + quadratic), φ̄ is unchanged (+0.30, p 0.085; size R² ≈ 0.01), so
     this is not a mean-level size shift. A spread-level size effect is not excluded.
  5. GAN: 1/36 grid settings at p ≤ 0.05 (K2 full), about chance; GAN∩Artifact (one origin, 11
     groups) also null.
  6. T2 (divergence vs date gap): ρ 0.08–0.27 in all four family×feature combinations, p ≈ 0.05
     only for diffusion φ̄ over all (origin-mixed) pairs. Same-origin pairs (6 diffusion, 22 GAN):
     ρ ≤ 0.21, p ≥ 0.19.
  7. Bin GMMs choose 1–2 components, aligned with neither generator (ARI ≤ 0.08) nor origin
     (≤ 0.02). Per generator, BIC picks 1 component for 13/16 diffusion and 10/13 GAN, which is
     uninformative at n = 17.
  8. Variables (exploratory): in diffusion, φ̄ rings 2 (+0.64) and 3+ (+0.48) and even offsets
     (+0.43) > ring 1 (+0.16); per-channel deviations +0.10.
- **Plain reading:** no demonstrated within-family time signal. For GANs, a large (≥ 2 SD) drift
  is ruled out. For diffusion, a weak, sign-consistent signal that is so far indistinguishable from
  the rise in output resolution. Nothing supports dating an unseen generator from Φ yet.
- **Caveats:** 17 images per generator; 12–13 release groups; φ̄ promoted after a pilot on this
  sample (its p is optimistic); the 7-day release window merges unrelated same-week releases.
- **Next:** the pre-registered spec (D-008/D-009) on the server's stratified-36k; a
  resolution-controlled arm (equal-resolution generators, or a common-pixel-scale crop policy as an
  explicitly separate departure from D-006).

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
