from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/"config"/"pipeline_contract.json").read_text(encoding="utf-8"))

def require(path):
    if not path.is_file():
        raise SystemExit(f"Missing required repository artifact: {path}")
    return path

def read_json(path):
    return json.loads(require(path).read_text(encoding="utf-8"))

def audit_all_pass(path, expected):
    df=pd.read_csv(require(path))
    if len(df)!=expected:
        raise SystemExit(f"{path}: expected {expected} gates, found {len(df)}")
    if "status" not in df.columns or not df["status"].astype(str).str.upper().eq("PASS").all():
        raise SystemExit(f"{path}: audit contains non-PASS gate")
    return df

s12=read_json(ROOT/"results"/"stage12"/"STAGE12_CANONICAL_FEATURE_MANIFEST.json")
s13=read_json(ROOT/"results"/"stage13"/"STAGE13_FINAL_MANIFEST.json")
s14=read_json(ROOT/"results"/"stage14"/"STAGE14_FINAL_MANIFEST.json")
s15=read_json(ROOT/"results"/"stage15"/"STAGE15_FINAL_MANIFEST.json")
s16=read_json(ROOT/"results"/"stage16"/"STAGE16_FINAL_MANIFEST.json")

audit_all_pass(ROOT/"results"/"stage12"/"STAGE12_FINAL_AUDIT.csv",20)
audit_all_pass(ROOT/"results"/"stage13"/"STAGE13_FINAL_AUDIT.csv",33)
audit_all_pass(ROOT/"results"/"stage14"/"STAGE14_FINAL_AUDIT.csv",28)
audit_all_pass(ROOT/"results"/"stage15"/"STAGE15_FINAL_AUDIT.csv",26)
audit_all_pass(ROOT/"results"/"stage16"/"STAGE16_FINAL_AUDIT.csv",36)

expected_model=CONTRACT["system"]
expected_sha=CONTRACT["protocol_sha"]
expected_fusion=CONTRACT["fusion_implementation_id"]
expected_dsc=CONTRACT["segmentation_metrics"]["DSC"]["mean"]

checks=[
    ("Stage12 COMPLETE",s12.get("status")=="COMPLETE"),
    ("Stage13 COMPLETE",s13.get("status")=="COMPLETE"),
    ("Stage14 COMPLETE",s14.get("status")=="COMPLETE"),
    ("Stage15 COMPLETE",s15.get("status")=="COMPLETE"),
    ("Stage16 COMPLETE",s16.get("status")=="COMPLETE"),
    ("Stage12 model",s12.get("segmentation_parent")==expected_model),
    ("Stage13 model",s13.get("segmentation_parent")==expected_model),
    ("Stage14 model",s14.get("segmentation_parent")==expected_model),
    ("Stage15 model",s15.get("final_model")==expected_model),
    ("Stage16 model",s16.get("final_model")==expected_model),
    ("Stage12 protocol",s12.get("protocol_sha")==expected_sha),
    ("Stage13 protocol",s13.get("protocol_sha")==expected_sha),
    ("Stage15 protocol",s15.get("protocol_sha")==expected_sha),
    ("Stage16 protocol",s16.get("protocol_sha")==expected_sha),
    ("Stage12 fusion",s12.get("fusion_implementation_id")==expected_fusion),
    ("Stage13 fusion",s13.get("fusion_implementation_id")==expected_fusion),
    ("Stage15 fusion",s15.get("fusion_implementation_id")==expected_fusion),
    ("Stage16 fusion",s16.get("fusion_implementation_id")==expected_fusion),
    ("Stage16 claim scope",s16.get("claim_scope")=="development five-fold CV"),
    ("Stage16 no external claim",s16.get("external_or_unseen_test_claim") is False),
]
bad=[name for name,ok in checks if not ok]
if bad:
    raise SystemExit("Repository verification failed: "+", ".join(bad))

dsc=float(s16["metrics"]["DSC"]["mean"])
if abs(dsc-expected_dsc)>5e-9:
    raise SystemExit(f"Stage16 DSC mismatch: {dsc} vs {expected_dsc}")

print("GRAPHMS REPOSITORY VERIFICATION PASS")
print("Stages 12-16 manifests: COMPLETE")
print("Audits: 20/20, 33/33, 28/28, 26/26, 36/36 PASS")
print(f"Final model: {expected_model}")
print(f"Development 5-fold DSC: {dsc:.12f}")
