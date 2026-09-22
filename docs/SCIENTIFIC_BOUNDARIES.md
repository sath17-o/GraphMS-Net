# Scientific boundaries

This repository preserves the frozen GraphMS v3.5.1 research result without expanding its claims.

- The segmentation result is a **development five-fold CV** result, not untouched external validation.
- Final equal-fold DSC is **0.748049553870**.
- The CNN reference is slightly higher overall; GraphMS v3.5.1 Hybrid is retained as the guide-aligned final hybrid system rather than being described as a universal numerical winner.
- Stage13 uses a classical SVM classifier and Ridge regressor. AdamW/cosine training settings belong to the neural segmentation lineage, not the classical models.
- Historical true multi-task learning was implemented/evaluated as an auxiliary branch and was not promoted.
- No new neural training is performed by this repository packaging layer.
- Stage16 risk evidence remains Stage13 evidence and is not invented as a sixth segmentation metric.
