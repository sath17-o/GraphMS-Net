# GraphMS-Net

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

## Full MRI runner implemented; CUDA acceptance pending

The patient runner now executes the frozen CNN → graph → GAT → Hybrid →
Stage11 → canonical Stage12 → Stage13 → local report path. It takes FLAIR/T1/T2,
restores masks to FLAIR geometry, and records input/asset/output hashes.

Import the original fold assets from your mounted project using
`scripts/setup_assets.py --source-root PATH --folds N`, then follow
[the run instructions](docs/EVALUATOR_RUN.md). No feature-bank download is
required for a new inference run. The importer creates a local SHA-256 lock;
no public release bundle is currently available.

Known development cases use their held-out fold. Unseen cases require explicit
fold selection and remain research runs. A fresh clone still needs the original
neural weights and a CUDA runtime. End-to-end GPU acceptance and exact saved-mask
comparison have **not yet been completed**.

CPU tests check verbatim v6 primitives, exact GAT/Hybrid forward parity,
asset tampering, fold protection, image geometry and canonical-feature-to-risk
execution. These tests do not substitute for the pending GPU acceptance run.

## Scientific boundaries

- The segmentation result is **development five-fold CV**, not untouched external validation.
- Historical true multi-task learning is **IMPLEMENTED_EVALUATED_AUXILIARY_NOT_PROMOTED**; the final segmentation and Stage13 risk outputs are linked but decoupled.
- Stage13 is classical SVM/Ridge inference; neural AdamW/cosine settings do not describe the classical models.
- No Gaussian-overlap replacement, GAT-Hybrid blend, TTA or dense-stride rescue is promoted.
- The repository does not invent an unvalidated five-fold ensemble rule for a new patient.
