"""One-command GraphMS-Net patient showcase.

This is a presentation-friendly entry point over the frozen patient runner.
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
    args = ap.parse_args()

    from graphms.patient import run_patient

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
