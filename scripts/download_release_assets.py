"""Download frozen GraphMS neural assets from the repository release.

Release assets are kept out of Git history because each ResEncM-250 checkpoint
is ~819 MB. Every downloaded file is size- and SHA-256-verified against the
release manifest before it is admitted to pretrained/assets.lock.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER_REPO = "sath17-o/GraphMS-Net"
DEFAULT_TAG = "assets-v1"
PROTOCOL = "8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3"
MANIFEST_NAME = "graphms-assets-v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def release_url(tag: str, name: str) -> str:
    return f"https://github.com/{OWNER_REPO}/releases/download/{tag}/{name}"


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            total = response.headers.get("Content-Length")
            copied = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                copied += len(block)
                if total:
                    print(f"  {target.name}: {copied / (1024**2):.1f} / {int(total) / (1024**2):.1f} MiB", end="\r", flush=True)
        print()
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def load_manifest(tag: str) -> dict:
    with urllib.request.urlopen(release_url(tag, MANIFEST_NAME), timeout=60) as r:
        manifest = json.load(r)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported release-asset manifest schema")
    if manifest.get("protocol_sha") != PROTOCOL:
        raise RuntimeError("Release-asset manifest protocol mismatch")
    return manifest


def ensure_release_assets(asset_root: Path, folds: list[int], tag: str = DEFAULT_TAG) -> dict:
    asset_root = Path(asset_root).expanduser().resolve()
    manifest = load_manifest(tag)
    records = manifest.get("files", {})
    needed = ["resencm250/plans.json", "resencm250/dataset.json"]
    for fold in sorted(set(folds)):
        if fold not in range(5):
            raise ValueError("Fold must be 0-4")
        needed.extend([
            f"resencm250/fold_{fold}/checkpoint_final.pth",
            f"gat/fold_{fold}/checkpoint_final.pth",
            f"fusion/fold_{fold}/checkpoint_final.pth",
        ])

    missing_manifest = [rel for rel in needed if rel not in records]
    if missing_manifest:
        raise RuntimeError(
            "The GitHub release does not contain the requested frozen assets: "
            + ", ".join(missing_manifest)
        )

    lock_path = asset_root / "assets.lock.json"
    lock = {
        "schema_version": 1,
        "protocol_sha": PROTOCOL,
        "trust": f"downloaded from GitHub release {tag} and SHA-256 verified",
        "release_tag": tag,
        "files": {},
    }
    if lock_path.is_file():
        old = json.loads(lock_path.read_text())
        if old.get("protocol_sha") == PROTOCOL:
            lock["files"].update(old.get("files", {}))

    for rel in needed:
        rec = records[rel]
        target = asset_root / rel
        good = (
            target.is_file()
            and target.stat().st_size == int(rec["size_bytes"])
            and sha256(target) == rec["sha256"]
        )
        if not good:
            print("Downloading", rel, flush=True)
            fetch(release_url(tag, rec["asset"]), target)
        if target.stat().st_size != int(rec["size_bytes"]) or sha256(target) != rec["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Release asset integrity mismatch: {rel}")
        lock["files"][rel] = {
            "sha256": rec["sha256"],
            "size_bytes": int(rec["size_bytes"]),
        }

    asset_root.mkdir(parents=True, exist_ok=True)
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2) + "\n")
    tmp.replace(lock_path)

    from graphms.assets import verify_assets
    for fold in folds:
        verify_assets(asset_root, fold)
    print("FROZEN RELEASE ASSETS VERIFIED:", ", ".join(map(str, folds)))
    return lock


def main() -> None:
    p = argparse.ArgumentParser(description="Download SHA-256-verified frozen GraphMS assets from GitHub Releases.")
    p.add_argument("--asset-root", type=Path, default=ROOT / "pretrained")
    p.add_argument("--folds", type=int, choices=range(5), nargs="+", default=[0])
    p.add_argument("--tag", default=DEFAULT_TAG)
    args = p.parse_args()
    ensure_release_assets(args.asset_root, args.folds, args.tag)


if __name__ == "__main__":
    main()
