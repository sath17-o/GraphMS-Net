# Runtime modes

GraphMS has two scientifically different reproducibility problems and they are kept separate.

## 1. Frozen-result verification/evaluation

`RUN_VERIFY.bat` and `RUN_FULL_EVALUATION.bat` run without neural retraining. They verify the Stage12-16 evidence and reproduce the final 93-case Stage16 aggregation.

## 2. Neural pipeline execution

The promoted neural chain is:

ResEncM-250 -> Stage5/6 graph feature construction -> Stage7 GAT -> Stage8 CNN+GNN concat + SE + self-attention + multi-scale fusion -> Stage9 decoder -> Stage11 cross-fitted post-processing.

The exact Stage7 and Stage8/9 geometries are packaged under `graphms/models/`.

The original five ResEncM-250 checkpoints and Stage5/6 feature banks are large binary artifacts. They must not be silently replaced by newly trained or approximate models. Until the release bundle is published and hash-registered, `RUN_GRAPHMS.bat` intentionally fails closed.

For a known development case, the correct outer-fold model and that fold's frozen Stage11 recipe must be used. The repository does not invent an unvalidated new-patient five-fold ensemble rule.
