"""Install the frozen detector panel on this machine, driven by panel.json.

    python3.12 detectors/install_panel.py              # install everything
    python3.12 detectors/install_panel.py --check      # verify an existing install, change nothing
    python3.12 detectors/install_panel.py --only aeroblade

Replaces the by-hand sequence in README.md's "Setup". The difference that matters is that this
checks out the *pinned* upstream commit from panel.json rather than whatever `--depth 1` happens
to fetch today, and verifies each weight's sha256 against the same file. A panel that is "frozen"
on one machine and "latest" on another is not a frozen panel, and the scores would not be
comparable across the two -- which is the entire premise of reusing one panel across experiments.

Platform handling: torch and torchvision come from PyTorch's own wheel index on Linux with an
NVIDIA driver, and from PyPI otherwise. The CUDA build is chosen from the *driver* version, not
from whatever is newest -- see cuda_channel().
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL_JSON = HERE / "panel.json"

# Where the virtualenvs live. They are ~4 GB each with a bundled CUDA runtime, and on the lab
# server / is a 492 GB volume shared with 18 other home directories while /data has terabytes.
# The repo gets a symlink, so detectors/<name>/.venv/bin/python works either way.
VENV_ROOT = Path(os.environ.get("PANEL_VENV_ROOT", "")) if os.environ.get("PANEL_VENV_ROOT") else None

# One directory per upstream clone; dmimagedetection's two panel members share one.
DIRS = {
    "cnndetection": "CNNDetection",
    "univfd": "UniversalFakeDetect",
    "dmimagedetection": "DMimageDetection",
    "aeroblade": "aeroblade",
}


def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def sha256_file(path, chunk=1 << 20):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def driver_cuda_version():
    """Highest CUDA runtime this NVIDIA driver supports, or None if there is no driver."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return match.group(1) if match else None


def cuda_channel():
    """Which PyTorch wheel index to use, decided by the installed driver.

    This is the one thing most likely to produce a panel that installs cleanly and then dies at
    the first .to("cuda"). PyPI's default torch==2.5.1 wheel is built against CUDA 12.4, which
    needs driver >= 525.60.13. sapucay runs 520.61.05, so it needs the cu118 build; installing
    the default and discovering that at hour three of an overnight run is the failure this
    avoids.
    """
    if platform.system() != "Linux":
        return None, "not Linux -- using PyPI wheels"
    version = driver_cuda_version()
    if version is None:
        return None, "no NVIDIA driver -- using PyPI (CPU) wheels"
    major, minor = (int(p) for p in version.split("."))
    if (major, minor) >= (12, 4):
        return "cu124", f"driver supports CUDA {version}"
    if major >= 12:
        return "cu121", f"driver supports CUDA {version}"
    return "cu118", f"driver supports CUDA {version} (< 12.4, so not the default PyPI wheel)"


def venv_path(name):
    return (VENV_ROOT / name) if VENV_ROOT else (HERE / name / ".venv")


def venv_python(name):
    return venv_path(name) / "bin" / "python"


def ensure_clone(name, repo, commit):
    """Clone if absent, then pin to the exact commit panel.json records."""
    dest = HERE / name / DIRS[name]
    if not dest.exists():
        print(f"[{name}] cloning {repo}")
        run(["git", "clone", "--filter=blob:none", repo, dest])
    have = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if have == commit:
        print(f"[{name}] already at pinned commit {commit[:12]}")
        return dest
    print(f"[{name}] checking out pinned {commit[:12]} (was {have[:12]})")
    # A --depth 1 clone will not have the pinned commit; fetch it specifically.
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit],
                   check=False, stderr=subprocess.DEVNULL)
    run(["git", "-C", str(dest), "checkout", "--quiet", commit])
    return dest


def ensure_venv(name, python):
    path = venv_path(name)
    link = HERE / name / ".venv"
    if not (path / "bin" / "python").exists():
        print(f"[{name}] creating venv at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        run([python, "-m", "venv", path])
    else:
        print(f"[{name}] venv exists at {path}")

    # Only when the venv lives outside the repo; otherwise path *is* the link target.
    if VENV_ROOT and not link.exists():
        link.symlink_to(path)
        print(f"[{name}] linked {link} -> {path}")

    run([venv_python(name), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "wheel"])


def install_requirements(name, channel):
    """torch/torchvision from the right wheel index first, then everything else from PyPI.

    Two steps rather than one requirements file per platform: with torch 2.5.1+cu118 already
    satisfying the `torch==2.5.1` line, pip leaves it alone, so the single pinned requirements.txt
    stays the one source of truth for both machines.
    """
    python = venv_python(name)
    requirements = HERE / name / "requirements.txt"
    pins = {}
    for line in requirements.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.startswith(("torch==", "torchvision==")):
            package, version = line.split("==")
            pins[package] = version

    if channel and pins:
        index = f"https://download.pytorch.org/whl/{channel}"
        print(f"[{name}] torch from {index}")
        run([python, "-m", "pip", "install", "--index-url", index,
             *[f"{p}=={v}" for p, v in pins.items()]])

    print(f"[{name}] requirements.txt")
    run([python, "-m", "pip", "install", "-r", requirements])

    if name == "aeroblade":
        # Upstream's own package, without its CUDA pip-freeze of a dependency list.
        run([python, "-m", "pip", "install", "-e", HERE / name / DIRS[name], "--no-deps"])


def fetch_weights(name):
    script = HERE / name / "fetch_weights.sh"
    if not script.exists():
        print(f"[{name}] no weights to fetch")
        return
    print(f"[{name}] fetching weights")
    run(["bash", script])


def verify(members, channel):
    """Check the things that silently produce wrong scores rather than errors."""
    ok = True
    for name in sorted(DIRS):
        python = venv_python(name)
        if not python.exists():
            print(f"  {name:<18} MISSING venv")
            ok = False
            continue
        probe = (
            "import torch, json;"
            "print(json.dumps({'torch': torch.__version__,"
            " 'cuda_build': torch.version.cuda,"
            " 'cuda_available': torch.cuda.is_available(),"
            " 'devices': torch.cuda.device_count()}))"
        )
        try:
            out = subprocess.check_output([str(python), "-c", probe], text=True,
                                          stderr=subprocess.STDOUT).strip()
            info = json.loads(out.splitlines()[-1])
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            print(f"  {name:<18} torch import FAILED: {error}")
            ok = False
            continue

        expected_cuda = bool(channel)
        status = "ok"
        if expected_cuda and not info["cuda_available"]:
            status = "NO CUDA (installed but not usable -- wrong wheel for this driver?)"
            ok = False
        print(f"  {name:<18} torch {info['torch']:<14} cuda_build={info['cuda_build']} "
              f"available={info['cuda_available']} devices={info['devices']}  {status}")

    print("\n  weights:")
    for member in members:
        for weight in member.get("weights", []):
            path = HERE / weight["path"]
            if not path.exists():
                print(f"    {weight['path']}  MISSING")
                ok = False
                continue
            digest = sha256_file(path)
            match = digest == weight["sha256"]
            print(f"    {weight['path']}  {'ok' if match else 'SHA MISMATCH'}")
            if not match:
                print(f"      expected {weight['sha256']}\n      got      {digest}")
                ok = False
    return ok


def main():
    global VENV_ROOT

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(DIRS), default=sorted(DIRS))
    parser.add_argument("--check", action="store_true", help="verify only; install nothing")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used to create the venvs (needs to be 3.12)")
    parser.add_argument("--venv-root", type=Path, default=VENV_ROOT,
                        help="put venvs here instead of detectors/<name>/.venv (they are ~4 GB each)")
    args = parser.parse_args()

    VENV_ROOT = args.venv_root

    panel = json.loads(PANEL_JSON.read_text())
    members = panel["members"]
    channel, why = cuda_channel()
    print(f"panel version {panel['panel_version']}, {len(members)} members")
    print(f"wheels:  {channel or 'PyPI default'}  ({why})")
    print(f"venvs:   {VENV_ROOT or HERE / '<name>' / '.venv'}\n")

    if args.check:
        sys.exit(0 if verify(members, channel) else 1)

    # One clone per directory even though dmimagedetection backs two panel members.
    by_dir = {}
    for member in members:
        name = next(d for d in DIRS if member["name"].startswith(d))
        by_dir.setdefault(name, member)

    for name in args.only:
        member = by_dir[name]
        print(f"\n=== {name} ===")
        ensure_clone(name, member["upstream_repo"], member["upstream_commit"])
        ensure_venv(name, args.python)
        install_requirements(name, channel)
        fetch_weights(name)

    print("\n=== verify ===")
    ok = verify(members, channel)
    print("\ndone." if ok else "\ndone, with problems above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
