from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
contract=json.loads((ROOT/"config"/"pipeline_contract.json").read_text(encoding="utf-8"))
per_case=pd.read_csv(ROOT/"results"/"stage16"/"STAGE16_PER_CASE_METRICS.csv")
metrics=["DSC","IoU","Sensitivity","Specificity","HD95_mm"]

if len(per_case)!=93 or per_case["case"].nunique()!=93:
    raise SystemExit("Expected exactly 93 unique Stage16 cases")
counts={str(int(k)):int(v) for k,v in per_case.groupby("fold").size().to_dict().items()}
if counts!=contract["fold_counts"]:
    raise SystemExit(f"Fold-count mismatch: {counts}")

per_fold=per_case.groupby("fold",as_index=False)[metrics].mean().sort_values("fold")
if per_fold["fold"].astype(int).tolist()!=[0,1,2,3,4]:
    raise SystemExit("Expected folds 0..4")

print("GraphMS v3.5.1 Stage16 evaluation replay")
for m in metrics:
    vals=per_fold[m].to_numpy(float)
    mean=float(vals.mean())
    sd=float(vals.std(ddof=1))
    ref=contract["segmentation_metrics"][m]
    if abs(mean-float(ref["mean"]))>5e-9:
        raise SystemExit(f"{m} mean mismatch: {mean} vs {ref['mean']}")
    if abs(sd-float(ref["sd"]))>5e-9:
        raise SystemExit(f"{m} SD mismatch: {sd} vs {ref['sd']}")
    print(f"{m:12s}: {mean:.12f} +/- {sd:.12f}")

print("EVALUATION REPLAY PASS")
print("Claim scope: DEVELOPMENT FIVE-FOLD CV")
