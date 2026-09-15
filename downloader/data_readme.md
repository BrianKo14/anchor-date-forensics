# AI-GenBench dataset — `/data/aigenbench`

**Owner:** `bkovo` · FCEN-UBA, Pablo Negri's group.
**Purpose:** Licenciatura thesis on time-calibrated AI-generated-image detection
(collaboration with GAMI Munich + TUM).

This is a local reconstruction of the **AI-GenBench v1.0.0** dataset
([MI-BioLab/AI-GenBench](https://github.com/MI-BioLab/AI-GenBench), arXiv 2504.20865):
180,000 authentic + 180,000 synthetic images across the benchmark's `train` and `validation`
splits, with one deliberate deviation (RAISE is subsampled — see below).

> **If you are not bkovo and you found this directory, or a long-running process writing to it,
> skip to [Sharing the machine](#sharing-the-machine) at the bottom.** Short version: it is a
> bandwidth-heavy but CPU-light download, it throttles itself when the machine gets busy, and you
> can pause it without destroying anything.

---

## Is it finished?

This directory fills over roughly a day. To find out where it is:

```bash
cd ~bkovo/anchor-date-forensics
./downloader/run.sh status            # is the job alive, and the tail of its log
.venv/bin/python -m downloader.verify # full integrity check (safe to run mid-download)
cat /data/aigenbench/manifest_meta.json
```

`manifest_meta.json` is the machine-readable record of what was actually built — row counts per
split, the data root, the RAISE target, and the column schema. Trust it over this file, which is
static prose and can drift.

## Layout

```
/data/aigenbench/
├── authentic/
│   ├── images/                        COCO + LAION + RAISE, normalized JPEG
│   ├── authentic_train.parquet        manifest (+ .jsonl, same rows, readable diffs)
│   └── authentic_validation.parquet
├── fakes/
│   ├── images/                        180,000 synthetic, normalized JPEG
│   ├── fakes_train.parquet
│   └── fakes_validation.parquet
├── cache/
│   ├── parquet/                       the 72 published AI-GenBench fake shards (~35 GB, kept)
│   ├── fake_meta/                     per-shard metadata sidecars, used to build the manifests
│   └── zips/                          COCO archives; deleted automatically after extraction
├── control/                           world-writable; see "Sharing the machine"
├── manifest_meta.json
└── README.md                          this file
```

Expect roughly **70 GB** when complete (~35 GB of that is the retained parquet shards, which can
be deleted if the space is ever needed — they are re-downloadable and only the JPEGs are used
downstream).

## What is in it

### Authentic half — 180,000 images

Not a sample we chose: AI-GenBench pins an exact list of file ids, shipped in its repo at
`dataset_creation/resources/{train,validation}_real_file_ids.txt`. We fetch exactly those.

| Source | train | validation | how it is fetched |
|---|---:|---:|---|
| COCO2017 train | 79,288 | 12,006 | the official zips, extracted locally |
| COCO2017 val | 3,796 | 68 | ditto |
| LAION-400M | 55,618 | 23,862 | per-URL from the benchmark's scraped filelist |
| **RAISE** | **subsampled** | **64** | per-URL TIFF from `193.205.194.113` |

**RAISE is the one deliberate departure from the benchmark.** The pinned selection is 5,362
images, but that host serves ~75 KB/s per connection and the TIFFs are 36.9 MB each — the full set
is ~198 GB of traffic and about a week of wall clock. We take **1,000** (all 64 validation ids plus
936 from train, drawn with a fixed seed after shuffling, since the catalogue is ordered by capture
session). That is sized for the thesis's central experiment — a false-positive rate on genuinely
digitized material — where 1,000 samples bound an observed 0% FPR at 95% CI [0, 0.37%].

LAION URLs rot; about half fail on any given pass. The benchmark ships 2.4× spare ids for exactly
this reason, and the downloader walks the shuffled spares until it hits the pinned count, retrying
transient failures (timeouts, dropped connections, 5xx) but not verdicts (404, 410, too small).

### Synthetic half — 180,000 images

From the published `lrzpellegrini/AI-GenBench-fake_part` parquet shards. **36 generators** spanning
2017-03 to 2024-08, uniformly sampled: 4,000 per generator in train, 1,000 in validation. The
generator registry and release dates come from the benchmark repo, not from us — they are the spine
of the thesis's G≤T construction and are deliberately not duplicated here.

---

## The invariant that matters

**Every image in both halves is re-encoded to JPEG quality 95 by a single function,
`imports/sample/imaging.py::prepare_image`.** Both halves share one quantization table.

This is not housekeeping. AI-GenBench's assembly ships with `make_jpeg_dataset = False`, which
leaves synthetic images in native containers (PNG, WEBP) while the authentic half arrives as web
JPEG. Written out that way, the two classes are separable on **container format alone** — a
detector would score near-perfectly while learning nothing about synthesis. That is the "Fake or
JPEG?" confound (Grommelt et al., arXiv 2403.17608), and it would invalidate every number computed
downstream.

> **If you add, replace, or re-encode any image in this directory, it must go through
> `prepare_image`, and you must re-run `downloader/verify.py` afterwards.** The verifier asserts
> that both halves carry exactly one shared quantization table and fails loudly otherwise. A
> silently mixed dataset looks fine and produces confidently wrong results.

This is also why the pipeline deliberately does **not** use `img2dataset`, which AI-GenBench's own
download scripts use: its `skip_reencode=True, encode_format="png"` path still re-encodes
everything to PNG, which would give the authentic class a different container from the synthetic
one.

## Manifests

Four parquet files (each mirrored as `.jsonl` so diffs stay readable), schema identical to the
612-image sample in the repo's `data/` directory:

```
file_id  origin_dataset  label  generator  description  width  height  path  split
                                    fakes also carry: source_format  release_date
```

- `label` — 0 authentic, 1 synthetic.
- `split` — the **benchmark's** train/validation partition, not an internal cut. (The small sample
  in the repo uses this column differently: everything there came from the benchmark's train split
  and was halved internally. Do not compare the two naively.)
- `path` — **relative to this directory**, not to the repo. The sample's manifests are repo-root
  relative; 70 GB cannot live in a git repo, so these are not. `manifest_meta.json` records which.
- `release_date` — from the benchmark's generator registry; load-bearing for the G≤T aggregation.

Consume them by pointing the existing tooling at this root:

```bash
export AIGENBENCH_DATA_ROOT=/data/aigenbench
```

`common.py` reads that variable and defaults to the in-repo `data/` without it, so the small sample
(on the `main` branch) and this full-dataset build coexist and nothing downstream needed changing.

## Provenance and licensing

Assembled from third-party datasets under their own terms. Nothing here was created by us; the
only transformation is the JPEG normalization described above.

- **AI-GenBench fake part** — `cc-by-nc-sa-4.0`. **Non-commercial.** Relevant if any of this ever
  leaves an academic context.
- **COCO 2017** — images are Flickr-sourced under their respective terms; see cocodataset.org.
- **LAION-400M** — the benchmark ships *URLs*, and the images are third-party web content fetched
  directly from origin hosts. LAION is an index, not a licence grant.
- **RAISE** — University of Trento, research use; see loki.disi.unitn.it/RAISE/.

Cite AI-GenBench (arXiv 2504.20865) for the dataset construction, not this directory.

## How it was built

Code lives in `~bkovo/anchor-date-forensics/downloader/` (version-controlled; the repo is
github.com/BrianKo14/anchor-date-forensics). Job state — a SQLite database of every file's
status — is in `~bkovo/anchor-date-forensics/var/`, deliberately outside this directory.

The download is fully resumable: images are written to `.part` and renamed, so a file that exists
here is complete by construction. Large archives resume from a byte offset. If the state database
were lost, `python -m downloader.pipeline --reconcile` rebuilds it by walking this tree — the
filesystem is the source of truth, not the database.

---

## Sharing the machine

**What the process is.** One Python process (`python -m downloader.pipeline`) under a tmux session
named `aigenbench`, owned by `bkovo`. It downloads a research dataset over roughly a day.

**What it costs.** It is **network- and disk-heavy, CPU-light**. It holds a global cap of ~12 MB/s
and a small number of connections per host. It touches **no GPU** — if you are waiting on a GPU,
this job is not why.

**It already yields to you.** A load governor samples `/proc/loadavg` every 15 s and halves its
thread pools above load 36, stopping fetches entirely above load 48 (on 48 cores). You should not
have to do anything for CPU contention.

**If it is in your way anyway** — most likely you want the *bandwidth* back, which the governor
does not manage:

```bash
touch /data/aigenbench/control/PAUSE      # job stands down within ~2 seconds
rm    /data/aigenbench/control/PAUSE      # and picks up exactly where it left off
```

That directory is world-writable for precisely this purpose. **Pausing destroys nothing** —
partial downloads are kept and the job resumes from the byte it stopped on. Please prefer this over
`kill`: killing it is also safe for the data, but it will not restart itself, and the run is
measured in days.

Any pause is recorded in the job's log with a timestamp, so it is visible rather than mysterious.

If something is genuinely wrong and you need it gone, `~bkovo/anchor-date-forensics/downloader/run.sh
stop` does a clean shutdown — but a note to `bkovo` is appreciated, since an unexplained stopped job
is indistinguishable from a crash.
