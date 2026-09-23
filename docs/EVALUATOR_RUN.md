# Run GraphMS-Net

The frozen result remains GraphMS v3.5.1 Hybrid, development five-fold CV,
DSC 0.748049553870. No training or model selection occurs in these commands.

## CPU verification

```sh
python -m pip install -r requirements-verify.txt
python scripts/run_pipeline.py --mode verify
python scripts/run_pipeline.py --mode evaluation-replay
```

The evaluation command aggregates the committed per-case table. It does not
rerun MRI inference or recompute voxel metrics from masks.

## Full MRI research inference: original GPU environment

Use a CUDA runtime (the original Colab/Kaggle runtime is appropriate). The
Windows laptop can view the generated HTML/PNG/NIfTI/CSV outputs. CPU execution
of the frozen mixed-precision neural path is not supported.

Install CUDA-enabled PyTorch compatible with your GPU/driver, then install the
remaining dependencies. The accepted training environment used PyTorch 2.8.0
with CUDA 12.6; do not install torch 2.9.x.

```sh
python -m pip install -r requirements-inference.txt
python scripts/check_neural_runtime.py
```

In Colab, first mount the original Drive account:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Import the frozen artifacts from your existing project. This does not require
public sharing, a release upload, feature-bank downloads, or retraining.
For a single development case, import only its held-out fold to save disk/I/O.

```sh
python scripts/setup_assets.py \
  --source-root /content/drive/MyDrive/MSLesSeg_MS \
  --folds 0
```

The importer accepts `MSLesSeg_MS` or its `nnunet_v2` child and reads:

- `nnUNet_results/Dataset001_MSLesSeg/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres/`
- `graphms_resencm250_true_hybrid_v3_5_1_8944f1a0de/outer_folds/outer_N/gat/refit/checkpoint_final.pth`
- `graphms_resencm250_true_hybrid_v3_5_1_8944f1a0de/outer_folds/outer_N/fusion/refit/checkpoint_final.pth`

It checks checkpoint identities, validates model tensors, verifies copies with
SHA-256 and records `pretrained/assets.lock.json`. This is a local attestation
of the explicitly selected originals, not a publisher-signed release. The
runner rechecks those hashes before loading checkpoints. Only use trusted
original checkpoints; their PyTorch serialization contains pickle objects.

Run the known fold-0 development case without loading its ground truth:

```sh
python scripts/run_pipeline.py --mode patient \
  --case-id MSLesSeg_P10_T1 \
  --flair /content/drive/MyDrive/MSLesSeg_MS/nnunet_v2/nnUNet_raw/Dataset001_MSLesSeg/imagesTr/MSLesSeg_P10_T1_0000.nii.gz \
  --t1 /content/drive/MyDrive/MSLesSeg_MS/nnunet_v2/nnUNet_raw/Dataset001_MSLesSeg/imagesTr/MSLesSeg_P10_T1_0001.nii.gz \
  --t2 /content/drive/MyDrive/MSLesSeg_MS/nnunet_v2/nnUNet_raw/Dataset001_MSLesSeg/imagesTr/MSLesSeg_P10_T1_0002.nii.gz \
  --output outputs/MSLesSeg_P10_T1
```

The canonical Harvard-Oxford atlas is fetched through Nilearn if not cached.
For an offline run, add `--atlas-cache PATH` pointing to the original Nilearn
atlas cache. Atlas loading must succeed before Stage13 reporting. The
canonical coordinate-compatibility and missing-value rules remain unchanged.

Inputs must be scalar, finite, nonempty 3-D NIfTI volumes, co-registered with
the same shape, spacing, origin and direction. The frozen nnU-Net plan performs
its original preprocessing. No new BET/N4/ANTs preprocessing is inserted.

For a case outside the 93-case development registry, explicitly pass `--fold N`.
Such a run is labelled an unseen-case selected-fold research run. There is no
automatic ensemble or new-patient validation claim. For a known development
case, the runner selects its held-out fold and rejects a conflicting fold.
Changing a case ID does not make a training patient unseen; preserve IDs.

## Outputs

A successful run creates the requested directory with:

- `lesion_probability.nii.gz` and `lesion_mask.nii.gz`, in input FLAIR geometry
- `features.csv` and `lesions.csv`, canonical Stage12 outputs
- `risk.json`, predictions from the committed refreshed Stage13 SVM/Ridge
- `overlay.png` and `patient_report.html`
- `provenance.json` and `COMPLETE.json`, recording inputs, asset lock and output hashes

An existing output directory is never overwritten. Failed runs remove their
incomplete temporary output directory and do not write a completion record.

## End-to-end CUDA acceptance

The implementation has CPU contract/parity tests and has now passed a one-case
end-to-end CUDA acceptance run with the original fold-0 artifacts for
`MSLesSeg_P10_T1`. The independently generated final mask matched the frozen
Stage12 reference geometry and binary mask exactly with **0 mismatched voxels**.
The committed record is
`evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json`.

This is one-case implementation/replay evidence. It is not an all-fold replay,
external validation, or a new clinical-performance claim. Public neural-asset
publication also remains separate from this acceptance result.

The full one-case acceptance command imports the correct fold, executes the
pipeline, then compares its mask against the saved frozen Stage12 mask:

```sh
python scripts/acceptance_replay.py \
  --project-root /content/drive/MyDrive/MSLesSeg_MS \
  --case-id MSLesSeg_P10_T1 \
  --output outputs/acceptance_P10_T1
```

It writes `ACCEPTANCE.json` with PASS/FAIL and mismatch count. The reference
mask is read only after the independent inference process has finished.
