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

### Frozen model assets

The evaluator does **not** need the original research Drive for neural model
assets. The public `assets-v1` GitHub Release is published and the fresh-clone
no-original-Drive evaluator path has passed. `run_graphms.py` first resolves the
held-out/selected fold, verifies any local assets, and if necessary downloads
only the required CNN/GAT/Hybrid fold plus the nnU-Net plan/dataset metadata.
The download is accepted only after size and SHA-256 checks against the release
manifest, then written into `pretrained/assets.lock.json`.

You can prefetch a fold explicitly:

```sh
python scripts/download_release_assets.py --folds 0
```

The owner-side fallback remains available for recovery or release creation:

```sh
python scripts/setup_assets.py \
  --source-root /content/drive/MyDrive/MSLesSeg_MS \
  --folds 0
```

The local importer and the release downloader both converge on the same
`pretrained/` layout and the same runtime `verify_assets` integrity gate.
Only trusted frozen checkpoints are loaded; PyTorch checkpoint serialization
contains pickle objects.

Run the known fold-0 development case without loading its ground truth:

```sh
python run_graphms.py \
  --case-id MSLesSeg_P10_T1 \
  --flair /path/to/MSLesSeg_P10_T1_0000.nii.gz \
  --t1 /path/to/MSLesSeg_P10_T1_0001.nii.gz \
  --t2 /path/to/MSLesSeg_P10_T1_0002.nii.gz \
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
external validation, or a new clinical-performance claim.

The repository has also passed a fresh-clone evaluator-path run in which the
frozen fold-0 neural assets were obtained from the public `assets-v1` GitHub
Release, verified locally, and the complete patient pipeline finished with
`GRAPHMS RESEARCH RUN COMPLETE` without using the original Drive model
workspace. That separate packaging/runtime acceptance is recorded in
`evidence/acceptance/NO_DRIVE_EVALUATOR_ACCEPTANCE.json`.

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
