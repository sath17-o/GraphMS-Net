# Segmentation Validation and Guide Alignment

This note isolates the segmentation portion of the prescribed GraphMS-Net framework and maps it to the frozen final implementation without changing the scientific result.

## Frozen final segmentation chain

`FLAIR + T1 + T2 → frozen nnU-Net/ResEncM preprocessing → ResEncM-250 → graph construction → TrueGAT → CNN/GNN hybrid fusion → transposed-convolution decoder → 1×1×1 lesion head → lesion probability → cross-fitted Stage11 → final lesion mask`

The final development result remains **DSC 0.748049553870 ± 0.031999993653** and **HD95 8.643581867636 ± 2.269264959846 mm** across five patient-level development folds. No external/unseen validation claim is made.

## Prescribed stages 2–11 versus frozen implementation

| Stage | Prescribed framework | Frozen final implementation | Status |
|---|---|---|---|
| 2 Preprocessing | BET, N4ITK, affine/ANTs, resampling, Z-score, 3-D patching | Frozen nnU-Net plan-based preprocessing used by the trained ResEncM-250 lineage | **Different implementation lineage; do not inject a second preprocessing chain at inference** |
| 3 Augmentation | rotation, flip, scale, elastic, gamma/noise, crop | training-time augmentation lineage | **Training evidence** |
| 4 Input fusion | FLAIR/T1/T2/PD channel concatenation | FLAIR/T1/T2; PD unavailable in the project dataset | **Dataset-constrained implementation** |
| 5 CNN | U-Net/ResNet/DenseNet family | ResEncM-250 residual nnU-Net backbone | **Implemented** |
| 6 Graph | SLIC/patch nodes; kNN/radius; Euclidean/cosine | 4³ grid nodes; 6-neighbour + spatial kNN8 + feature kNN4 + self-loops | **Implemented using prescribed patch/kNN family** |
| 7 GNN | GCN/GAT/message passing | two-layer edge-aware TrueGAT, hidden 128, four heads | **Implemented** |
| 8 Hybrid fusion | concatenation, SE/self-attention, multi-scale fusion | CNN/GNN concat + SE + self-attention + multi-scale fusion | **Implemented** |
| 9 Segmentation head | decoder, skip connections, 1×1 output, sigmoid | transposed-convolution decoder + CNN-logit skip + 1×1×1 residual lesion head | **Implemented** |
| 10 Loss | Dice, BCE, optional Focal | Dice + BCE; Focal not promoted | **Implemented** |
| 11 Post-processing | threshold, connected components, morphology, small-lesion removal | cross-fitted threshold + 26-CCA + remove components <10 voxels; morphology none | **Implemented; morphology not promoted** |

## Binary probability clarification

The final head uses two logits and converts them to lesion probability. For logits `[b, b+d]`, the lesion softmax probability is:

`exp(b+d) / (exp(b) + exp(b+d)) = exp(d) / (1 + exp(d)) = sigmoid(d)`.

Therefore the frozen two-logit formulation is binary-probability equivalent to applying a sigmoid to the lesion-vs-background logit difference; it is not a missing activation.

## Stage11 status

The frozen recipes are:

- folds 0, 1, 3, 4: probability threshold 0.40
- fold 2: probability threshold 0.45
- connectivity: 26
- minimum connected-component size: 10 voxels
- morphology: none

Earlier development artifacts include morphology/filtering experiments, but those artifacts are not relabeled as final Hybrid evidence. The final Hybrid Stage11 remains the leakage-controlled recipe that produced the committed Stage16 table.

## Reviewer-facing qualitative audit

Run `notebooks/GraphMS_Segmentation_Audit.ipynb` in Colab with the existing project Drive mounted. It performs **no training and no tuning**. It deterministically selects the best, median, and worst cases from the committed 93-case Stage16 DSC table and renders:

**FLAIR | expert ground truth | frozen final prediction | TP/FP/FN error map**

with case-level DSC, IoU, sensitivity, specificity and HD95.

This makes segmentation failure modes directly inspectable while preserving the frozen scientific result.

## Submission boundary

Do not add BET/N4/ANTs only at inference and continue quoting the existing Stage16 metrics. Such a change would alter the input distribution of a model trained and evaluated under the frozen nnU-Net preprocessing lineage. A literal BET/N4/ANTs replacement would require a separately trained/evaluated experiment and is intentionally outside this rapid no-retraining audit.


## Cohort-level diagnostic extension

The no-retraining audit now also computes case-level DSC/HD95 distributions, fold-wise DSC, lesion-burden quartiles, the lowest-DSC table, FN/FP-dominant error direction, and an exportable diagnostics bundle. The frozen 93-case record shows a residual small-lesion sensitivity weakness; the detailed values and interpretation are recorded in `docs/SEGMENTATION_COHORT_DIAGNOSTICS.md`.
