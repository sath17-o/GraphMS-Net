# Evaluator run guide

The repository is being packaged in two layers.

## Available now

### Verify frozen scientific evidence

```bat
RUN_VERIFY.bat
```

This checks the frozen Stage12-16 manifests and requires audits of 20/20, 33/33, 28/28, 26/26 and 36/36 PASS.

### Replay the final Stage16 metric aggregation

```bat
RUN_FULL_EVALUATION.bat
```

This recomputes the five outer-fold means and sample standard deviations from the frozen 93-case per-case evaluation table and requires exact agreement with the Stage16 contract.

## Heavyweight patient inference

`RUN_GRAPHMS.bat` is intentionally fail-closed until the inference-only ResEncM-250, GAT and Hybrid/Fusion checkpoint bundle is published with SHA-256 values. The repository will not silently substitute a different model or claim a patient-inference path before those assets are frozen.
