from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from graphms.stage13 import predict_stage13

df=pd.read_csv(ROOT/"precomputed"/"stage12"/"stage13_DEV_ONLY_93_crosssectional.csv")
if len(df)!=93 or df["case_id"].nunique()!=93:
    raise SystemExit("Frozen Stage12->13 handoff must contain exactly 93 unique cases")
row=df.iloc[[0]].copy()
out=predict_stage13(
    row,
    ROOT/"pretrained"/"stage13"/"stage13_primary_edss_ge4_classifier.joblib",
    ROOT/"pretrained"/"stage13"/"stage13_primary_edss_regressor.joblib",
)
print("GRAPHMS STAGE13 LOCAL INFERENCE SMOKE PASS")
print("case_id:",row.iloc[0]["case_id"])
for k,v in out.items():
    print(f"{k}: {v}")
