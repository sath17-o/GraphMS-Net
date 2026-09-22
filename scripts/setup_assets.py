from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
reg=json.loads((ROOT/"config"/"asset_registry.json").read_text(encoding="utf-8"))
pending=[x for x in reg["required_for_patient_inference"] if x["status"].startswith("PENDING")]
if pending:
    print("GraphMS heavyweight inference assets are still being packaged.")
    for item in pending:
        print(" -",item["id"],":",item["status"])
    raise SystemExit(
        "RUN_GRAPHMS is intentionally fail-closed until the inference-only "
        "checkpoint bundle and SHA-256 registry are published."
    )
print("ASSET SETUP PASS")
