# `lr/` — time-calibrated likelihood ratios

The calibration and aggregation layers `CLAUDE.md` specifies (architecture items 3 and 4), as
importable pieces so `lr_time_calibrated.ipynb` can stay a narrative rather than a code dump.

Nothing here re-runs a detector. The input is `scores/*.csv`, which `./run_all.sh` produced.

| module | what it holds |
|---|---|
| `data.py` | load the scored panel, audit its provenance, σ-rescale without leakage, stratified folds |
| `core.py` | the 50/50 construction — PAV and logistic calibrators, ridge fusion |
| `metrics.py` | Cllr and its decomposition, Tippett, DET, rates of misleading evidence |
| `bounds.py` | ELUB, plus the count and source-count cross-checks |
| `models.py` | per-family score models (fusion → calibration → bounds), truncated and per-generator variants, cross-fitting |
| `aggregate.py` | max over `G≤T`, monotonicity checks, the crossing point `T*(F)` |
| `report.py` | the cross-tabulations this experiment reports |
| `plots.py` | the figures |
| `selftest.py` | controls the rest of the pipeline depends on |

```python
import lr
M, deltas = lr.sigma_rescale(lr.load_scored_manifest())
fam = lr.family_registry(M)
tr, va = M[M.split == "train"], M[M.split == "val"]
mods = lr.fit_all(tr, list(fam.index), "per_family", l2s)
```

## The three pieces of algebra everything rests on

### 1. Why the output carries no prevalence assumption

Weight every H_p (fake) row `0.5/n_p` and every H_d (authentic) row `0.5/n_d`, so each class
carries total weight ½. An estimator fitted on that sample estimates the posterior of a
population whose class prior is exactly ½:

```
p̂(s) = ½f_p / (½f_p + ½f_d) = f_p / (f_p + f_d)   ⇒   p̂/(1−p̂) = f_p/f_d = LR(s)
```

Prior odds are 1, so posterior odds *are* the likelihood ratio. The only prior in the output is
the one the trier of fact supplies. `core.balanced_weights` is the whole implementation.

Only the weight **ratio** matters, so sklearn's `class_weight="balanced"` is the same
construction — but sklearn does **not** rescale its penalty by the weights, so `core.fuse`
normalises weights to sum to `n`. Without that, a given λ would regularise a 137-fake family and
an 8-fake family differently and the per-family λ values would not be comparable.

### 2. Why bounding is mandatory rather than cosmetic

A calibrated LR satisfies `E[LR | H_d] = 1`. For 50/50-weighted PAV this **provably fails
in-sample**, by a knowable amount. A PAV block `b` holding `n_pb` fakes and `n_db` authentics gets
`LR_b = (n_pb/n_p)/(n_db/n_d)`, so

```
E[LR | H_d] = Σ_b (n_db/n_d)·LR_b = Σ_{b: n_db>0} n_pb/n_p = 1 − P̂(LR = ∞ | H_p)
```

and symmetrically `E[1/LR | H_p] = 1 − P̂(LR = 0 | H_d)`. Fake mass landing in a block with no
authentics escapes into an `LR = ∞` atom and the expectation falls short by exactly that mass.
Unbounded PAV output is therefore not a likelihood ratio. `metrics.check_pav_identity` asserts
both identities to 1e-9.

### 3. Where ELUB comes from

Markov's inequality applied to the LR itself gives Royall's bound:

```
P(LR ≥ τ | H_d) = ∫_{LR≥τ} f_d = ∫_{LR≥τ} f_p/LR ≤ (1/τ)∫f_p ≤ 1/τ
```

ELUB is the empirical enforcement of exactly that, expressed as a normalised Bayes error rate. A
threshold τ is equivalent to prior odds `π = 1/(1+τ)`; against the neutral system whose error is
`min(π, 1−π)`,

```
NBE(τ) = [P̂(LR ≤ τ | H_p) + τ·P̂(LR > τ | H_d)] / min(1, τ)
```

A pair `(l, u)` is **admissible** iff the LRs clipped to `[l, u]`, augmented with one
maximally-misleading observation per class (an H_p case at `l`, an H_d case at `u`), satisfies
`NBE(τ) ≤ 1` for every τ. ELUB is the widest admissible pair; `(1, 1)` is always admissible, which
is what a system with nothing supportable returns.

`bounds.elub` scans a log grid and refines by bisection. It **asserts** monotonicity of NBE in each
bound rather than assuming it — NBE is empirically monotone here, but that is an observation, not a
theorem, and the two effects of widening a bound (more false alarms, fewer misses) pull opposite
ways. Sanity anchor: for perfectly separated classes it saturates at `count_bound(n_p, n_d)`.

**Attribution [R]:** the NBE criterion and the devil's-advocate augmentation are recalled from
Vergeer et al. (2016), *Sci. Justice*, and the `lir` package's `bounding.py`. They were not
re-read against the paper. The mathematics above is Markov's inequality and stands on its own; the
*citation* should be verified before it goes in the thesis.

## Two design points that are correctness requirements, not preferences

**Fusion must be per-family** (`models.py`). PAV output is monotone by construction, so if every
family's model were `φ_g(s)` for the *same* global scalar `s`, then `max_g LR_g` would itself be a
monotone function of `s`: the `G≤T` aggregator would add no information, `T*(F)` would be a
relabelling of one number, and family attribution would be impossible. Global fusion is kept only
as a comparator, and the notebook measures the collapse it causes.

**H_d for `LR_g` is the authentic class only** (`models.family_xy`). Other families' fakes are
neither hypothesis for that model — the question is "family g versus authentic" — and folding them
into the denominator collapses the bounds. The assertion is in the code because it is an easy and
quiet mistake.

## Bounds: three of them, and the choice is unsettled

`bounds` offers `elub`, `count_bound` and `source_bound`; `adopt_bounds` takes the tightest.

`source_bound(N) = [1/N, N]` with `N` the number of **generators** in the family — the unit of
analysis `temporal_correlation.ipynb` §5 established for this repo (272 diffusion images are 16
observations, not 272). It is the tightest bound on this sample, and for a single-generator family
it is `[1, 1]`: nothing supportable, at any detector quality.

The notebook reports both and does not settle which should be primary. The brief asked for ELUB, so
ELUB is primary there; on the measured evidence the source bound has the stronger claim — it brings
`E[max | authentic]` from 4.21 to 1.95 against a required 1.0, and roughly halves false crossings —
at the cost of a system that can never report exculpatory evidence once a `[1, 1]` family is in
`G≤T`.
