# Stage Implementation Matrix and Same-Cohort Development Context

This document summarizes the finalized GraphMS-Net stage-level implementation evidence and same-cohort development comparisons without changing the selected model, Stage11 recipes, Stage12/13 outputs, or Stage16 scientific results.

## Same-cohort development comparison

The historical branch comparison uses **series/case-weighted mean DSC**. The final Hybrid is therefore shown here using its corresponding 93-case arithmetic mean for direct comparison. The primary GraphMS Stage16 result remains **0.748049553870 ± 0.031999993653**, defined as the equal-weight mean ± sample SD of the five outer-fold case means.

| Branch | Evaluation scope | DSC | Disposition |
|---|---|---:|---|
| ResEncM-250 raw | 93-series, five-fold development OOF | 0.749353 | completed CNN reference |
| ResEncM-250 + component filter | 93-series, five-fold development OOF | 0.749416 | completed CNN reference |
| ResEncM-250 + filter + opening | 93-series, five-fold development OOF | 0.725885 | morphology evaluated; worse |
| CATMIL M150 raw | 93-series, five-fold development OOF | 0.744290 | completed auxiliary |
| Cross-fit calibrated CNN | 93-series, five-fold development OOF | 0.745780 | completed auxiliary |
| Stage7 GAT final | 93-series nested development OOF | 0.744531 | evaluated; not promoted |
| GraphMS v3.5.1 Hybrid | 93-case Stage16 OOF; case-weighted descriptive mean | 0.749349 | final selected graph-aware system |

**Interpretation:** the final GraphMS Hybrid is not presented as a numerical winner over the CNN reference. The filtered ResEncM-250 branch is marginally higher in this historical case-weighted Dice table. The Hybrid is retained as the selected graph-aware architecture.

**Morphology disposition:** opening was evaluated in the ResEncM reference lineage and reduced mean Dice from **0.749416** to **0.725885**. The final Hybrid therefore retains the cross-fitted Stage11 threshold + 26-connected-component analysis + minimum 10-voxel component rule with no morphology.

## 16-stage implementation matrix

| Stage | Pipeline component | Final implementation | Status |
|---:|---|---|---|
| 1 | Data acquisition | MSLesSeg development cohort; FLAIR/T1/T2 and expert lesion masks | implemented / finalized |
| 2 | Preprocessing | Co-registered multimodal inputs with finalized nnU-Net plan-based runtime preprocessing and geometry validation | implemented / finalized |
| 3 | Data augmentation | rotation / flip / scale / elastic / gamma-noise / crop | implemented — training only |
| 4 | Multimodal fusion | FLAIR + T1 + T2 channel-wise input; PD unavailable | implemented / dataset-constrained |
| 5 | CNN feature extraction | ResEncM-250 residual nnU-Net backbone + multiscale features | implemented |
| 6 | Graph construction | 4³ patch/grid nodes; 6-neighbour + spatial kNN8 + feature kNN4 + self-loops | implemented |
| 7 | GNN learning | two-layer edge-aware TrueGAT, hidden width 128, four heads | implemented |
| 8 | Hybrid fusion | CNN/GNN concatenation + SE + self-attention + multiscale fusion | implemented |
| 9 | Segmentation head | transposed-convolution decoder + skips + 1×1×1 lesion output | implemented |
| 10 | Segmentation loss | Dice + BCE | implemented; Focal not promoted |
| 11 | Post-processing | cross-fitted threshold + 26-CCA + remove components <10 voxels | implemented; morphology evaluated / not promoted |
| 12 | Lesion features | volume/count/shape + GLCM + multimodal/spatial context | implemented — Stage12 v2.1 |
| 13 | Risk prediction | MRI_SPATIAL_SVM + MRI_SPATIAL_RIDGE from finalized Stage12 imaging features | implemented research output |
| 14 | Multi-task learning | historical joint Dice+BCE + CE branch | implemented / evaluated / auxiliary / not promoted |
| 15 | Model training | AdamW + CosineAnnealingLR + dropout 0.30 + weight decay 1e-4 | completed provenance |
| 16 | Evaluation | DSC/IoU/sensitivity/specificity/HD95/precision/lesion-F1 + downstream risk metrics | complete / finalized |

All 16 stages are traceable to implemented or evaluated project evidence. Extensions that require genuinely new experiments—new pretraining or transfer learning, hard-negative or small-lesion-aware training, and independent external validation—remain future research directions rather than being retroactively inferred from the finalized development evidence.

## Extended Stage16 segmentation metrics

The finalized 93-case Stage16 masks were re-evaluated without retraining, new neural inference, threshold tuning, or model selection to report the remaining segmentation-detection metrics.

| Metric | Equal-fold mean ± sample SD |
|---|---:|
| Segmentation precision (case-macro) | 0.777737 ± 0.039188 |
| Lesion F1 — any overlap | 0.744368 ± 0.029950 |
| Lesion F1 — IoU ≥ 0.10 | 0.730296 ± 0.033212 |

Precision is computed per case and averaged within each outer fold. Lesion F1 uses 26-connected components and maximum bipartite one-to-one matching, with either any overlap or component-pair IoU ≥ 0.10. These are development five-fold descriptive metrics from the same finalized masks as the primary Stage16 result.
