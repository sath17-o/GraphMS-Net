# Guide Closure — Final No-Retraining Alignment

This note closes the remaining low-cost alignment items from the supplied GraphMS-Net guide without changing the frozen model, Stage11 recipes, Stage12/13 outputs, or Stage16 scientific result.

## Same-cohort development comparison

The historical branch scoreboard uses **series/case-weighted mean DSC**. The final Hybrid is therefore shown here using its corresponding 93-case arithmetic mean for comparison. The primary official GraphMS Stage16 result remains **0.748049553870 ± 0.031999993653**, the equal-weight mean ± sample SD of the five outer-fold case means.

| Branch | Evaluation scope | DSC | Disposition |
|---|---|---:|---|
| ResEncM-250 raw | 93-series, five-fold development OOF | 0.749353 | completed CNN reference |
| ResEncM-250 + component filter | 93-series, five-fold development OOF | 0.749416 | completed CNN reference |
| ResEncM-250 + filter + opening | 93-series, five-fold development OOF | 0.725885 | morphology evaluated; worse |
| CATMIL M150 raw | 93-series, five-fold development OOF | 0.744290 | completed auxiliary |
| Cross-fit calibrated CNN | 93-series, five-fold development OOF | 0.745780 | completed auxiliary |
| Stage7 GAT final | 93-series nested development OOF | 0.744531 | evaluated; not promoted |
| GraphMS v3.5.1 Hybrid | 93-case frozen Stage16 OOF; case-weighted descriptive mean | 0.749349 | final selected graph-aware system |

**Interpretation:** the final GraphMS Hybrid is not presented as a numerical winner over the CNN reference. The filtered ResEncM-250 branch is marginally higher in this historical case-weighted Dice table. The Hybrid is retained as the selected graph-aware architecture.

**Morphology closure:** opening was explicitly evaluated in the older ResEncM reference lineage and reduced mean Dice from **0.749416** to **0.725885**. The final Hybrid therefore retains its leakage-controlled cross-fitted Stage11 threshold + 26-CCA + minimum 10-voxel component rule with no morphology.

## Prescribed 16-stage disposition

| Stage | Guide component | Final project disposition | Closure |
|---:|---|---|---|
| 1 | Data acquisition | Project MRI dataset + expert lesion masks | active / frozen evidence |
| 2 | Preprocessing | Prescribed BET/N4/registration lineage documented; frozen nnU-Net plan-based runtime preprocessing retained | documented boundary |
| 3 | Data augmentation | rotation / flip / scale / elastic / gamma-noise / crop | implemented — training only |
| 4 | Multimodal fusion | FLAIR + T1 + T2 channel-wise input; PD unavailable | active / dataset-constrained |
| 5 | CNN feature extraction | ResEncM-250 residual nnU-Net backbone + multiscale features | active |
| 6 | Graph construction | 4³ patch/grid nodes; 6-neighbour + spatial kNN8 + feature kNN4 + self-loops | active — patch/kNN family |
| 7 | GNN learning | two-layer edge-aware TrueGAT, hidden 128, four heads | active |
| 8 | Hybrid fusion | CNN/GNN concat + SE + self-attention + multiscale fusion | active |
| 9 | Segmentation head | transposed-convolution decoder + skips + 1×1×1 lesion output | active |
| 10 | Segmentation loss | Dice + BCE | active; Focal optional / not promoted |
| 11 | Post-processing | cross-fitted threshold + 26-CCA + remove components <10 voxels | active; morphology evaluated / not promoted |
| 12 | Lesion features | volume/count/shape + GLCM + multimodal/spatial context | active — Stage12 v2.1 |
| 13 | Risk prediction | MRI_SPATIAL_SVM + MRI_SPATIAL_RIDGE from frozen Stage12 imaging features | active research output |
| 14 | Multi-task learning | historical joint Dice+BCE + CE branch | implemented / evaluated / auxiliary / not promoted |
| 15 | Model training | AdamW + CosineAnnealingLR + dropout 0.30 + weight decay 1e-4 | completed provenance |
| 16 | Evaluation | segmentation DSC/IoU/sensitivity/specificity/HD95 + risk accuracy/precision/recall/F1/ROC-AUC | complete / frozen |

## Closure boundary

All 16 prescribed stages are traceable. Remaining guide ideas that would require genuinely new experiments—new pretraining/transfer learning, new hard-negative or small-lesion-aware training, or new external validation—are outside the frozen submission and are not simulated, backfilled, or claimed as completed.
