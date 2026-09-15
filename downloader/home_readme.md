# What is this long-running job? — `bkovo`

If you found a `python -m downloader.pipeline` process, a tmux session named **`aigenbench`**, or
heavy network traffic from this account: it is an academic dataset download, and this file explains
it. Full documentation lives in **`/data/aigenbench/README.md`**.

**Who / why.** `bkovo` — Licenciatura thesis on time-calibrated AI-generated-image detection, in
Pablo Negri's group (collaboration with GAMI Munich + TUM). It is reconstructing the AI-GenBench
benchmark: 360,000 images into `/data/aigenbench`, roughly 70 GB, over about a day.

**What it costs you.**

| Resource | Impact |
|---|---|
| **GPU** | **None.** It never touches CUDA. If you are queueing for a GPU, this is not the cause. |
| CPU | Light. Image re-encoding only, and it halves its own thread pools above load 36 and stops fetching above load 48. |
| Network | This is the real cost — capped at ~12 MB/s, a few connections per host. |
| Disk | On `/data` (1.5 TB free), never on `/`. |

**If it is in your way**, take the bandwidth back without breaking anything:

```bash
touch /data/aigenbench/control/PAUSE     # stands down in ~2s, keeps partial downloads
rm    /data/aigenbench/control/PAUSE     # resumes from exactly where it stopped
```

That directory is world-writable on purpose. Pausing is safe and logged. Please prefer it to
`kill` — killing is also safe for the data, but the job will not restart itself and the run takes
days. A clean shutdown is `~bkovo/anchor-date-forensics/downloader/run.sh stop`.

If you stop it, a note to `bkovo` is appreciated — a stopped job looks exactly like a crashed one.

---

*(For bkovo: `./downloader/run.sh status | attach | stop | start`; dashboard via
`ssh -L 8765:127.0.0.1:8765` then http://localhost:8765.)*
