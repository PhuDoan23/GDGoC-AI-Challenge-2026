"""
Pack the PPO agent into a competition-ready submission.zip.

Usage:
    python scripts/pack_submission.py --checkpoint ckpts/warmup_actor_final.pth
    python scripts/pack_submission.py --checkpoint ckpts/league/actor_5000000.pth --out submission_v2.zip
"""

import sys
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PPO_DIR = ROOT / "ppo_agent"

REQUIRED_FILES = ["agent.py", "encode_obs.py", "model.py"]


def pack(checkpoint: str, out: str):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_path = Path(out)
    tmp_dir = ROOT / "_submission_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # Copy required source files
    for fname in REQUIRED_FILES:
        src = PPO_DIR / fname
        if not src.exists():
            print(f"ERROR: missing source file: {src}")
            sys.exit(1)
        shutil.copy(src, tmp_dir / fname)

    # Copy checkpoint as model.pth
    shutil.copy(ckpt_path, tmp_dir / "model.pth")

    # Create ZIP
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in tmp_dir.iterdir():
            zf.write(f, f.name)

    shutil.rmtree(tmp_dir)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Submission packed: {out_path}  ({size_mb:.1f} MB)")

    # Sanity checks
    if size_mb > 100:
        print("WARNING: ZIP exceeds 100 MB limit!")
    ckpt_mb = ckpt_path.stat().st_size / (1024 * 1024)
    if ckpt_mb > 150:
        print("WARNING: checkpoint exceeds 150 MB single-file limit!")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="submission.zip")
    args = p.parse_args()
    pack(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
