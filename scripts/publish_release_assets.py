"""One-time publisher for the frozen GraphMS neural release assets.

Run this only from a trusted machine/Colab session that can access the original
MSLesSeg_MS project and is authenticated to GitHub with the `gh` CLI.
No model is retrained or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3"
EXPERIMENT = "graphms_resencm250_true_hybrid_v3_5_1_8944f1a0de"
CNN_DIR = "nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres"
TAG = "assets-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*cmd: str) -> None:
    subprocess.run(list(cmd), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--folds", type=int, choices=range(5), nargs="+", default=list(range(5)))
    p.add_argument("--tag", default=TAG)
    args = p.parse_args()

    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI 'gh' is required. Install it, then run 'gh auth login'.")
    run("gh", "auth", "status")

    source = args.source_root.expanduser().resolve()
    nnroot = source / "nnunet_v2" if (source / "nnunet_v2").is_dir() else source
    cnn = nnroot / "nnUNet_results" / "Dataset001_MSLesSeg" / CNN_DIR
    downstream = nnroot / EXPERIMENT / "outer_folds"

    from graphms.assets import validate_checkpoint

    pairs: list[tuple[Path, str, str]] = [
        (cnn / "plans.json", "resencm250/plans.json", "resencm250-plans.json"),
        (cnn / "dataset.json", "resencm250/dataset.json", "resencm250-dataset.json"),
    ]
    for fold in sorted(set(args.folds)):
        pairs += [
            (cnn / f"fold_{fold}/checkpoint_final.pth",
             f"resencm250/fold_{fold}/checkpoint_final.pth",
             f"resencm250-fold{fold}-checkpoint_final.pth"),
            (downstream / f"outer_{fold}/gat/refit/checkpoint_final.pth",
             f"gat/fold_{fold}/checkpoint_final.pth",
             f"gat-fold{fold}-checkpoint_final.pth"),
            (downstream / f"outer_{fold}/fusion/refit/checkpoint_final.pth",
             f"fusion/fold_{fold}/checkpoint_final.pth",
             f"fusion-fold{fold}-checkpoint_final.pth"),
        ]

    missing = [str(src) for src, _, _ in pairs if not src.is_file()]
    if missing:
        raise FileNotFoundError("Missing original frozen assets:\n" + "\n".join(missing))

    # Validate scientific identity before publication.
    dataset = json.loads((cnn / "dataset.json").read_text())
    channels = dataset.get("channel_names", {})
    if [str(channels.get(str(i), "")).lower() for i in range(3)] != ["flair", "t1", "t2"]:
        raise RuntimeError("Unexpected CNN channel order")
    plans = json.loads((cnn / "plans.json").read_text())
    if plans.get("plans_name") != "nnUNetResEncUNetMPlans":
        raise RuntimeError("Unexpected nnU-Net plan")

    records = {}
    for src, rel, asset_name in pairs:
        if rel.endswith(".pth"):
            kind = "resencm250" if rel.startswith("resencm250/") else ("gat" if rel.startswith("gat/") else "fusion")
            validate_checkpoint(src, kind)
        records[rel] = {
            "asset": asset_name,
            "sha256": sha256(src),
            "size_bytes": src.stat().st_size,
        }
        print(rel, records[rel]["sha256"], records[rel]["size_bytes"], flush=True)

    manifest = {
        "schema_version": 1,
        "release_tag": args.tag,
        "protocol_sha": PROTOCOL,
        "system": "GraphMS v3.5.1 Hybrid",
        "folds": sorted(set(args.folds)),
        "files": records,
    }

    with tempfile.TemporaryDirectory(prefix="graphms-release-") as td:
        td = Path(td)
        manifest_path = td / "graphms-assets-v1.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        # Create the release once; later calls can add/replace assets.
        exists = subprocess.run(
            ["gh", "release", "view", args.tag],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
        if not exists:
            run(
                "gh", "release", "create", args.tag,
                "--target", "main",
                "--title", "GraphMS-Net frozen neural assets v1",
                "--notes",
                "Frozen GraphMS v3.5.1 Hybrid inference weights. "
                "Scientific claim scope remains development five-fold CV."
            )

        for src, _, asset_name in pairs:
            # Stage and upload one file at a time. This keeps temporary disk
            # usage bounded by the largest checkpoint (~819 MB) and ensures
            # the release asset has the stable evaluator-facing filename.
            staged = td / asset_name
            staged.unlink(missing_ok=True)
            shutil.copyfile(src, staged)
            try:
                run("gh", "release", "upload", args.tag, str(staged), "--clobber")
            finally:
                staged.unlink(missing_ok=True)

        run("gh", "release", "upload", args.tag, str(manifest_path), "--clobber")

    print("PUBLISHED", args.tag, "folds", sorted(set(args.folds)))


if __name__ == "__main__":
    main()
