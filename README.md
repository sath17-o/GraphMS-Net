# GraphMS-Net

**Graph-aware multimodal 3D MRI lesion segmentation and EDSS-risk research pipeline**

> Packaging status: evaluator-ready repository construction in progress. The scientific pipeline is frozen through Stage 16; this repository is being assembled from the frozen GraphMS v3.5.1 artifacts without retraining or changing model selection.

## Frozen scientific identity

- Final segmentation system: **GraphMS v3.5.1 Hybrid**
- Protocol SHA: `8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3`
- Fusion implementation: `v3.5-concat-se-self-multiscale-dice-bce-adamw-cosine-250iter`
- Final Stage11: cross-fitted Stage11-v1
- Development 5-fold equal-fold DSC: **0.748049553870**
- Claim scope: **development five-fold CV**
- Current EDSS classifier: **MRI_SPATIAL_SVM, C=30**
- Current EDSS regressor: **MRI_SPATIAL_RIDGE, alpha=30**

The repository will expose separate paths for:

1. **verification** of frozen artifacts and scientific contracts;
2. **full evaluation replay** from frozen Stage11 masks;
3. **true GraphMS inference/demo execution** through CNN → graph → GAT → hybrid fusion → segmentation → Stage12 features → Stage13 risk;
4. **research replay** for the expensive fold-specific pipeline when the required model assets are installed.

## Important scientific boundaries

The final GraphMS v3.5.1 Hybrid development result is not presented as untouched external validation. Historical multi-task work is retained as auxiliary evidence and is **not** represented as the current final model having been jointly retrained with Stage13. Stage13 SVM/Ridge models are classical downstream models and are not attributed the AdamW/cosine training configuration used by the neural segmentation lineage.

This README will be expanded as the executable packaging layer is completed.
