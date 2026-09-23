# GraphMS-Net

**Complete guide-pipeline showcase: FLAIR + T1 + T2 in → segmentation + lesion analysis + EDSS/risk research outputs + report out**

GraphMS-Net now exposes the frozen patient path as a single end-to-end application. For the complete module-by-module map, see [PIPELINE_SHOWCASE.md](PIPELINE_SHOWCASE.md).

## One-command patient showcase

The evaluator-facing runner is designed to bootstrap the required frozen neural
fold automatically from the versioned GitHub Release when the weights are not
already present locally. Every downloaded asset is size- and SHA-256-verified
before inference. During release publication/testing, the original
`scripts/setup_assets.py --source-root PATH` path remains available as the
trusted fallback.

After installing the inference environment and validating CUDA:

```sh
python run_graphms.py \
  --case-id PATIENT_ID \
  --flair /path/to/FLAIR.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t2 /path/to/T2.nii.gz \
  --fold 0 \
  --output outputs/PATIENT_ID
```

For a known development case, omit `--fold`: the held-out fold is selected automatically. For an unseen case, an explicit fold is required because no new-patient ensemble rule was validated.

The single command executes the selected frozen path:

```text
FLAIR + T1 + T2
 -> validation / frozen preprocessing
 -> ResEncM-250 CNN
 -> Stage5/6 graph construction
 -> Stage7 TrueGAT
 -> GraphMS v3.5.1 Hybrid (CNN+GNN + SE + self-attention + multi-scale fusion)
 -> lesion probability/head
 -> cross-fitted Stage11
 -> final lesion mask
 -> canonical Stage12 v2.1
 -> Stage13 SVM/Ridge
 -> NIfTI + CSV + JSON + overlay + HTML report + provenance
```

A successful run creates `lesion_probability.nii.gz`, `lesion_mask.nii.gz`, `features.csv`, `lesions.csv`, `risk.json`, `overlay.png`, `patient_report.html`, `provenance.json` and `COMPLETE.json`.

### Executed CUDA acceptance

The packaged end-to-end path has now been executed on CUDA for development case `MSLesSeg_P10_T1` (held-out fold 0). It reproduced the frozen reference geometry and binary mask exactly: **PASS, 0 mismatched voxels**. The committed record is [evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json](evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json). Scope remains **one-case CUDA acceptance; not all-fold or external validation**.

**Graph-aware multimodal 3-D MRI lesion segmentation and EDSS-risk research pipeline**

This repository packages the frozen **GraphMS v3.5.1 Hybrid** research system as an evaluator-facing reproducibility project. The scientific result is frozen; repository work does not retrain or re-select the model.

## Frozen scientific identity

- Final segmentation system: **GraphMS v3.5.1 Hybrid**
- Protocol SHA: `8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3`
- Fusion implementation: `v3.5-concat-se-self-multiscale-dice-bce-adamw-cosine-250iter`
- Final Stage11: **cross-fitted Stage11-v1**, 26-connectivity, no morphology
- Development five-fold equal-fold DSC: **0.748049553870**
- Claim scope: **development five-fold CV**
- Stage13 classifier: **MRI_SPATIAL_SVM, C=30, mri_spatial**
- Stage13 regressor: **MRI_SPATIAL_RIDGE, alpha=30, mri_spatial**

## What works from a normal clone now

### 1. Frozen package verification

```bat
RUN_VERIFY.bat
```

Requires Stage12-16 audits of **20/20, 33/33, 28/28, 26/26 and 36/36 PASS** and verifies the frozen model/protocol identities.

### 2. Final Stage16 evaluation replay

```bat
RUN_FULL_EVALUATION.bat
```

Recomputes the five-fold aggregation from the frozen 93-case per-case evaluation table and requires exact agreement with:

- DSC: `0.748049553870 +/- 0.031999993653`
- IoU: `0.610534701872 +/- 0.036904187531`
- Sensitivity: `0.745722782031 +/- 0.061037864849`
- Specificity: `0.999688903970 +/- 0.000205231021`
- HD95: `8.643581867636 +/- 2.269264959846 mm`

### 3. Frozen Stage13 local inference smoke

```bat
RUN_STAGE13_SMOKE.bat
```

Loads the actual committed refreshed SVM/Ridge joblib artifacts and executes a local prediction against the frozen Stage12->13 development handoff.

## Neural source now packaged

The repository contains the frozen downstream implementation geometry:

```text
ResEncM-250
  -> Stage5/6 graph features
  -> Stage7 TrueGAT
  -> CNN+GNN concat
  -> SE
  -> self-attention
  -> multi-scale fusion
  -> transposed-convolution decoder
  -> CNN-logit skip
  -> 1x1x1 lesion head
  -> cross-fitted Stage11
  -> canonical Stage12 v2.1
  -> frozen Stage13 SVM/Ridge
```

- `graphms/models/gat.py`: audited Stage7 sparse edge-GAT geometry.
- `graphms/models/hybrid.py`: promoted SE + self-attention + multi-scale Stage8/9 geometry.
- `graphms/postprocessing.py`: exact cross-fitted Stage11-v1 recipes.
- `graphms/stage12/canonical_v2_1_source.py`: verbatim canonical Stage12 scientific engine source.
- `graphms/stage13.py`: local wrapper around the committed refreshed Stage13 artifacts.

## Full MRI runner implemented; one-case CUDA acceptance passed

The patient runner now executes the frozen CNN → graph → GAT → Hybrid →
Stage11 → canonical Stage12 → Stage13 → local report path. It takes FLAIR/T1/T2,
restores masks to FLAIR geometry, and records input/asset/output hashes.

The final distribution path uses the `assets-v1` GitHub Release. On first
patient execution, `run_graphms.py` resolves the required fold, downloads
only that fold's frozen CNN/GAT/Hybrid assets plus nnU-Net metadata, verifies
them against `graphms-assets-v1.json`, and creates `pretrained/assets.lock.json`.
The original Drive importer remains available for owner-side recovery and
publication. See [ASSET_PUBLICATION.md](ASSET_PUBLICATION.md).

Known development cases use their held-out fold. Unseen cases require explicit
fold selection and remain research runs. A fresh clone needs a CUDA runtime;
once the `assets-v1` release is published, the required neural weights are
retrieved automatically rather than requiring access to the original Drive. A one-case end-to-end CUDA acceptance and exact saved-mask comparison have **passed** for `MSLesSeg_P10_T1` (fold 0), with matching geometry and **0 mismatched voxels**. This is implementation/replay evidence only; it is not all-fold or external validation.

CPU tests check verbatim v6 primitives, exact GAT/Hybrid forward parity,
asset tampering, fold protection, image geometry and canonical-feature-to-risk
execution. These tests complement, but do not replace, the committed one-case CUDA acceptance evidence.

## Scientific boundaries

- The segmentation result is **development five-fold CV**, not untouched external validation.
- Historical true multi-task learning is **IMPLEMENTED_EVALUATED_AUXILIARY_NOT_PROMOTED**; the final segmentation and Stage13 risk outputs are linked but decoupled.
- Stage13 is classical SVM/Ridge inference; neural AdamW/cosine settings do not describe the classical models.
- No Gaussian-overlap replacement, GAT-Hybrid blend, TTA or dense-stride rescue is promoted.
- The repository does not invent an unvalidated five-fold ensemble rule for a new patient.
