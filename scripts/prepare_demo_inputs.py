"""Prepare the bundled GraphMS evaluator demo MRI triplet.

The public P10_T1 demo volumes are included under demo_inputs/. Larger files are
stored as byte-exact chunks to keep repository API writes reliable. This script
reassembles them and verifies SHA-256 before use.

No ground-truth lesion mask or patient metadata is bundled.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo_inputs"
PARTS = DEMO / "parts"

EXPECTED = {
    "MSLesSeg_P10_T1_0000.nii.gz": "77ec63aacb16a548f276efcac9dd2075897e976af56f6af458ba4f5343da04e5",
    "MSLesSeg_P10_T1_0001.nii.gz": "81bb4d304929acd753d2b9068c6d7c9360905ec466c8667d16679b212122071d",
    "MSLesSeg_P10_T1_0002.nii.gz": "3e74b01ced35e58dee539267e1ed81bf33f3ac6ec60445442e0595166065e805",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assemble(name: str, output_dir: Path) -> Path:
    dst = output_dir / name
    direct = DEMO / name
    if direct.exists():
        shutil.copyfile(direct, dst)
    else:
        chunks = sorted(PARTS.glob(name + ".part*"))
        if not chunks:
            raise FileNotFoundError(f"No bundled source or chunks found for {name}")
        with dst.open("wb") as out:
            for chunk in chunks:
                with chunk.open("rb") as src:
                    shutil.copyfileobj(src, out)

    got = sha256(dst)
    expected = EXPECTED[name]
    if got != expected:
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"Bundled demo integrity failure for {name}: expected {expected}, got {got}"
        )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        p = assemble(name, args.output_dir)
        print(f"DEMO INPUT VERIFIED: {p.name} {p.stat().st_size} bytes {EXPECTED[name]}")


if __name__ == "__main__":
    main()
