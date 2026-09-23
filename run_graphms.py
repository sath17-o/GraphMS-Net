"""GraphMS-Net patient-level frozen inference entry point.

This command exposes the frozen patient runner through a reproducible research interface.
It does not change scientific/model logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the complete frozen GraphMS-Net patient pipeline: "
            "FLAIR/T1/T2 -> CNN -> graph -> GAT -> Hybrid -> Stage11 -> "
            "Stage12 -> Stage13 -> visualization/report."
        )
    )
    ap.add_argument("--flair", type=Path, required=True, help="Co-registered FLAIR NIfTI")
    ap.add_argument("--t1", type=Path, required=True, help="Co-registered T1 NIfTI")
    ap.add_argument("--t2", type=Path, required=True, help="Co-registered T2 NIfTI")
    ap.add_argument("--case-id", required=True, help="Patient/case identifier")
    ap.add_argument("--fold", type=int, choices=range(5), help="Required only for cases outside the frozen 93-case registry")
    ap.add_argument("--asset-root", type=Path, default=ROOT / "pretrained")
    ap.add_argument("--atlas-cache", type=Path, default=ROOT / "pretrained" / "atlas_cache")
    ap.add_argument("--output", type=Path, required=True, help="New output directory")
    ap.add_argument(
        "--no-auto-assets",
        action="store_true",
        help="Do not download missing frozen neural assets from the GraphMS-Net GitHub Release.",
    )
    args = ap.parse_args()

    from graphms.assets import verify_assets
    from graphms.patient import resolve_fold, run_patient
    selected_fold, _ = resolve_fold(args.case_id, args.fold, ROOT)

    try:
        verify_assets(args.asset_root, selected_fold)
        print(f"Frozen neural assets verified locally for fold {selected_fold}.")
    except (RuntimeError, FileNotFoundError, ValueError) as local_error:
        if args.no_auto_assets:
            raise RuntimeError(
                f"Frozen fold-{selected_fold} assets are not usable locally and "
                "--no-auto-assets was requested."
            ) from local_error
        print(
            f"Frozen fold-{selected_fold} assets are missing or not verified locally; "
            "downloading the SHA-256-locked GitHub Release assets..."
        )
        try:
            from graphms.release_assets import ensure_release_assets
            ensure_release_assets(args.asset_root, [selected_fold])
        except Exception as release_error:
            raise RuntimeError(
                "Automatic frozen-asset bootstrap failed. If the public release has "
                "not been published yet, use scripts/setup_assets.py with the trusted "
                "original MSLesSeg_MS workspace, or publish assets-v1 first."
            ) from release_error

    print("=" * 72)
    print("GraphMS-Net v3.5.1 Hybrid — COMPLETE PATIENT PIPELINE")
    print("FLAIR + T1 + T2")
    print("  -> frozen nnU-Net preprocessing / ResEncM-250")
    print("  -> Stage5/6 graph features")
    print("  -> Stage7 TrueGAT")
    print("  -> CNN+GNN Hybrid fusion (SE + self-attention + multi-scale)")
    print("  -> lesion head")
    print("  -> cross-fitted Stage11")
    print("  -> canonical Stage12 v2.1")
    print("  -> frozen Stage13 SVM/Ridge")
    print("  -> overlay + HTML report + provenance")
    print("=" * 72)
    run_patient(args, ROOT)


if __name__ == "__main__":
    main()
