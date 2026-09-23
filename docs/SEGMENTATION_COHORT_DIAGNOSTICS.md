# Segmentation Cohort Diagnostics

This document summarizes a **no-retraining, no-retuning** diagnostic analysis of the frozen GraphMS v3.5.1 Hybrid Stage16 segmentation record.

## Scope

- 93 development cases across five patient-level outer folds.
- No new neural training, inference, threshold search, morphology search, or model selection.
- Ground truth is used only for retrospective evaluation and failure analysis.
- The primary frozen result remains the equal-weight mean of the five outer-fold case means: **DSC 0.748049553870 ± 0.031999993653** and **HD95 8.643581867636 ± 2.269264959846 mm**.
- These diagnostics do not create an external/unseen validation claim.

## Case-level descriptive findings

| Diagnostic | Frozen Stage16 observation |
|---|---:|
| Case-level DSC median | 0.772401 |
| Case-level DSC range | 0.325253–0.921168 |
| DSC < 0.50 | 4 / 93 |
| DSC < 0.60 | 9 / 93 |
| DSC ≥ 0.80 | 40 / 93 |
| HD95 > 20 mm | 15 / 93 |
| Sensitivity < 0.50 | 6 / 93 |
| FN-dominant among DSC < 0.60 | 7 / 9 |

The low-DSC tail is therefore predominantly **false-negative (missed-lesion) dominated**, rather than being driven only by excess false positives.

## Lesion-burden diagnostic

Expert lesion burden was computed retrospectively as `TP + FN` from the committed OOF confusion counts. It is not an inference feature.

| Expert lesion-burden quartile | n | Burden range, voxels | Mean DSC | Median DSC |
|---|---:|---:|---:|---:|
| Q1 — smallest | 23 | 744–2,843 | 0.6573 | 0.6578 |
| Q2 | 23 | 2,857–5,807 | 0.7129 | 0.7294 |
| Q3 | 23 | 6,217–17,068 | 0.7813 | 0.7803 |
| Q4 — largest | 24 | 17,877–72,872 | 0.8419 | 0.8561 |

The descriptive Pearson correlation between `log10(expert lesion voxels + 1)` and DSC is **r = 0.6242**. This supports a development-set failure-mode interpretation that smaller/subtler lesion burden is a residual weakness of the frozen system. It does **not** establish causation or external clinical generalization.

## Representative frozen cases

The companion notebook selects cases deterministically from the committed DSC column:

| Selection | Case | Fold | DSC | IoU | Sensitivity | HD95 mm |
|---|---|---:|---:|---:|---:|---:|
| Best | MSLesSeg_P4_T3 | 3 | 0.921168 | 0.853857 | 0.955659 | 1.00 |
| Median | MSLesSeg_P6_T1 | 1 | 0.772401 | 0.629197 | 0.787749 | 2.24 |
| Worst | MSLesSeg_P18_T1 | 3 | 0.325253 | 0.194210 | 0.209091 | 21.29 |

For each case the notebook renders **FLAIR | expert ground truth | frozen final prediction | TP/FP/FN error map**. The worst case is deliberately retained rather than hidden.

## Submission interpretation

The new diagnostic evidence strengthens the segmentation section without creating a new scientific system. The frozen checkpoints, nnU-Net preprocessing lineage, GraphMS Hybrid architecture, cross-fitted Stage11 recipes, and Stage16 headline metrics remain unchanged.

The residual weakness is stated directly: **case performance is heterogeneous, and the lower-DSC tail is concentrated in lower lesion burden with predominantly missed-lesion error.**

Notebook: `notebooks/GraphMS_Segmentation_Audit.ipynb`
