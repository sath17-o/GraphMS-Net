# GraphMS-Net

**Graph-aware multimodal 3-D MRI research pipeline for multiple-sclerosis lesion segmentation, lesion characterization, and downstream MRI-derived EDSS/risk modeling**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sath17-o/GraphMS-Net/blob/main/notebooks/GraphMS_Evaluator_Colab.ipynb)
[![Verify frozen GraphMS package](https://github.com/sath17-o/GraphMS-Net/actions/workflows/verify.yml/badge.svg)](https://github.com/sath17-o/GraphMS-Net/actions/workflows/verify.yml)

GraphMS-Net packages the frozen **GraphMS v3.5.1 Hybrid** research system as a reproducible inference and evaluation repository. The repository preserves the selected scientific pipeline, its associated implementation identities, downstream models, evaluation artifacts, and provenance checks. Repository packaging does **not** retrain, re-select, or modify the frozen scientific result.

The primary research claim remains **development five-fold cross-validation**. This repository does not present the reported performance as independent external validation or as evidence of clinical deployment readiness.


## Research results at a glance

| Research item | Frozen result |
|---|---:|
| Final segmentation system | **GraphMS v3.5.1 Hybrid** |
| Development cohort | **93 MRI scans / 53 patients** |
| Validation design | **5-fold development cross-validation** |
| DSC | **0.748049553870 ± 0.031999993653** |
| IoU / Jaccard | **0.610534701872 ± 0.036904187531** |
| Sensitivity | **0.745722782031 ± 0.061037864849** |
| Specificity | **0.999688903970 ± 0.000205231021** |
| HD95 | **8.643581867636 ± 2.269264959846 mm** |
| Stage13 fixed-OOF ROC-AUC | **0.747453703704** |
| Stage13 fixed-OOF RMSE | **1.671117674945** |
| Stage12–16 audits | **20/20, 33/33, 28/28, 26/26, 36/36 PASS** |
| Exact CUDA mask replay | **PASS — 0 mismatched voxels** |
| Public-release reproducibility execution | **PASS** |
| Claim scope | **development five-fold CV; not external clinical validation** |

The Colab notebook expands this summary into the complete Stage13 and Stage16 metric record, per-fold and per-case segmentation results, methodological/audit evidence, and a ground-truth-free patient-level reproducibility run.

## Reproducibility notebook

The recommended entry point is the Colab reproducibility notebook:

[**Open GraphMS-Net Reproducibility Notebook in Google Colab**](https://colab.research.google.com/github/sath17-o/GraphMS-Net/blob/main/notebooks/GraphMS_Evaluator_Colab.ipynb)

Notebook source: [`notebooks/GraphMS_Evaluator_Colab.ipynb`](notebooks/GraphMS_Evaluator_Colab.ipynb)

The default notebook execution is **zero-upload**. It reconstructs the attributed `MSLesSeg_P10_T1` FLAIR/T1/T2 demonstration triplet bundled with the repository, verifies the three inputs by SHA-256, verifies the frozen research package, and presents the complete committed quantitative research record before patient-level inference. This includes Stage16 aggregate, per-fold, and all 93 case-level segmentation metrics; the complete frozen Stage13 classification and regression metric sets; Stage11 post-processing recipes; Stage12 integrity; Stage14 task status; Stage15 training provenance; Stage12–16 audit counts; and implementation-acceptance evidence. It then resolves the appropriate development fold, downloads and verifies the corresponding frozen neural assets from the public `assets-v1` GitHub Release, executes the complete patient-level pipeline, displays the segmentation and downstream outputs, verifies provenance, and packages the generated results.

For an independent MRI case, set `USE_BUNDLED_DEMO = False` in the notebook and provide a co-registered FLAIR/T1/T2 NIfTI triplet. Cases outside the frozen development registry require an explicitly selected fold because no new-patient five-fold ensemble rule was validated.

## Frozen scientific identity

| Component | Frozen specification |
|---|---|
| Final segmentation system | **GraphMS v3.5.1 Hybrid** |
| Protocol SHA | `8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3` |
| Stage5/6 runtime implementation | `v6-stage56-hardened-provenance-reuse-validate-20260919` |
| Stage7 implementation | `v3.4.1-gat-fp32-dice-bce-adamw-cosine-baseline-preserving-selection` |
| Hybrid fusion implementation | `v3.5-concat-se-self-multiscale-dice-bce-adamw-cosine-250iter` |
| Final Stage11 | cross-fitted Stage11-v1; 26-connectivity; no morphology |
| Stage13 classifier | MRI_SPATIAL_SVM, C=30, `mri_spatial` |
| Stage13 regressor | MRI_SPATIAL_RIDGE, alpha=30, `mri_spatial` |
| Stage14 status | `IMPLEMENTED_EVALUATED_AUXILIARY_NOT_PROMOTED` |
| Scientific claim scope | development five-fold cross-validation |

## Frozen inference architecture

The patient-level runtime executes the selected frozen path:

```text
FLAIR + T1 + T2
    ↓
input and geometry validation
    ↓
frozen nnU-Net preprocessing
    ↓
ResEncM-250 CNN
    ↓
Stage5/6 graph-feature construction
    ↓
Stage7 TrueGAT
    ↓
CNN + GNN concatenation
    ↓
SE + self-attention + multi-scale hybrid fusion
    ↓
transposed-convolution decoder + CNN-logit skip
    ↓
1×1×1 lesion head
    ↓
cross-fitted Stage11-v1
    ↓
final lesion mask in input FLAIR geometry
    ↓
canonical Stage12 v2.1 lesion/MRI-spatial features
    ↓
frozen Stage13 SVM/Ridge
    ↓
NIfTI + CSV + JSON + overlay + HTML report + provenance
```

Detailed module-to-implementation correspondence is provided in [`ARCHITECTURE_AND_REPRODUCIBILITY.md`](ARCHITECTURE_AND_REPRODUCIBILITY.md).

## Patient-level inference

After installing the inference environment and confirming a CUDA-capable runtime:

```bash
python run_graphms.py \
  --case-id PATIENT_ID \
  --flair /path/to/FLAIR.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t2 /path/to/T2.nii.gz \
  --fold 0 \
  --output outputs/PATIENT_ID
```

For a known development case, omit `--fold`; the held-out fold is resolved from the frozen registry. For a case outside that registry, an explicit fold `0..4` is required. Such a run is labeled as a selected-fold research run and does not constitute external validation.

On first use, `run_graphms.py` automatically retrieves the required frozen ResEncM-250, GAT, and Hybrid checkpoints together with the nnU-Net metadata from the public `assets-v1` release. Asset size and SHA-256 identity are verified before the files are admitted into `pretrained/assets.lock.json`.

A successful execution generates:

| Output | Description |
|---|---|
| `lesion_probability.nii.gz` | voxelwise lesion probability volume |
| `lesion_mask.nii.gz` | final binary lesion segmentation in input FLAIR geometry |
| `features.csv` | canonical Stage12 feature table |
| `lesions.csv` | lesion-component measurements |
| `risk.json` | frozen Stage13 research outputs |
| `overlay.png` | segmentation visualization |
| `patient_report.html` | consolidated patient-level research report |
| `provenance.json` | input, asset, execution, and output provenance |
| `COMPLETE.json` | explicit successful-run completion record |

## Development evaluation result

The frozen Stage16 five-fold aggregate is:

| Metric | Mean ± SD |
|---|---:|
| DSC | **0.748049553870 ± 0.031999993653** |
| IoU | **0.610534701872 ± 0.036904187531** |
| Sensitivity | **0.745722782031 ± 0.061037864849** |
| Specificity | **0.999688903970 ± 0.000205231021** |
| HD95 | **8.643581867636 ± 2.269264959846 mm** |

These values are reproduced from the frozen **93-case development five-fold evaluation table**. They are not presented as independent external-test performance.

The aggregate can be replayed locally with:

```bash
python scripts/run_pipeline.py --mode evaluation-replay
```

## Reproducibility and acceptance evidence

The repository includes several distinct forms of implementation evidence.

| Evidence | Status | Scope |
|---|---|---|
| Frozen manifests and Stage12-16 audits | PASS | repository/package integrity |
| CPU source-parity and inference-contract tests | PASS | implementation contracts and frozen primitive parity |
| Stage13 local inference smoke test | PASS | committed SVM/Ridge runtime |
| `MSLesSeg_P10_T1` CUDA end-to-end replay | PASS | one development case, held-out fold 0 |
| Exact final-mask comparison for `MSLesSeg_P10_T1` | PASS | matching geometry; 0 mismatched voxels |
| Fresh-clone public-release execution without original model Drive | PASS | packaging/runtime reproducibility |

Committed acceptance records:

- [`evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json`](evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json)
- [`evidence/acceptance/NO_DRIVE_EVALUATOR_ACCEPTANCE.json`](evidence/acceptance/NO_DRIVE_EVALUATOR_ACCEPTANCE.json)

These records demonstrate implementation and packaging reproducibility. They do not extend the scientific claim beyond development five-fold cross-validation.

## Local verification

Repository-level verification:

```bash
python -m pip install -r requirements-verify.txt
python scripts/run_pipeline.py --mode verify
```

CUDA inference-runtime verification:

```bash
python -m pip install -r requirements-inference.txt
python scripts/check_neural_runtime.py
```

The accepted neural environment used PyTorch 2.8.0 with CUDA 12.6 and `nnunetv2==2.8.1`. The full mixed-precision 3-D neural path requires CUDA; package verification and Stage16 aggregate replay remain CPU-capable.

## Demonstration MRI data

The zero-upload Colab workflow uses only the three MRI volumes required for the `MSLesSeg_P10_T1` reproducibility demonstration. No ground-truth lesion mask, EDSS record, or additional patient metadata is bundled.

Source:

Ali M. Muslim, *Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information*, Mendeley Data, Version 1 (2022), DOI: `10.17632/8bctsm8jz7.1`.

The source dataset is distributed under **CC BY 4.0**. Attribution and reconstruction details are documented in [`demo_inputs/README.md`](demo_inputs/README.md). The demonstration files are verified by SHA-256 before use.

## Repository structure

```text
GraphMS-Net/
├── graphms/                  frozen runtime implementation
├── config/                   protocol and asset registries
├── results/                  frozen Stage12-16 research artifacts
├── evidence/acceptance/      committed reproducibility evidence
├── pretrained/               local verified asset layout
├── demo_inputs/              attributed reproducibility MRI triplet
├── notebooks/                Colab reproducibility notebook
├── scripts/                  verification, replay, asset, and inference utilities
├── tests/                    frozen source-parity and inference-contract tests
├── run_graphms.py            patient-level inference entry point
├── ARCHITECTURE_AND_REPRODUCIBILITY.md  frozen architecture and implementation map
└── docs/EVALUATOR_RUN.md     detailed execution protocol
```

## Scope and limitations

GraphMS-Net is a research reproducibility package. The current repository should be interpreted under the following constraints:

1. Reported segmentation performance is based on **development five-fold cross-validation**, not untouched external validation.
2. The one-case CUDA acceptance is an implementation/replay check; it is not an all-fold validation study.
3. The Stage14 true multi-task branch was implemented and evaluated but was **not promoted** into the final segmentation system.
4. Stage13 uses frozen classical SVM/Ridge models; neural training settings do not describe those downstream estimators.
5. No unvalidated five-fold ensemble rule is introduced for a new patient. External cases require an explicitly selected fold and remain research runs.
6. No additional BET/N4/ANTs preprocessing is inserted into patient inference; the runtime uses the frozen nnU-Net preprocessing plan.
7. The repository and generated outputs are intended for research and reproducibility, not clinical diagnosis or treatment decision-making.

## Documentation

For implementation-level detail, see:

- [`ARCHITECTURE_AND_REPRODUCIBILITY.md`](ARCHITECTURE_AND_REPRODUCIBILITY.md) — module-to-implementation correspondence and runtime architecture
- [`docs/EVALUATOR_RUN.md`](docs/EVALUATOR_RUN.md) — detailed execution protocol
- [`ASSET_PUBLICATION.md`](ASSET_PUBLICATION.md) — frozen neural-asset publication and integrity workflow
- [`config/asset_registry.json`](config/asset_registry.json) — frozen asset identities and publication state
