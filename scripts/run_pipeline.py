from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    subprocess.run(cmd, check=True)

def verify():
    run([sys.executable, str(ROOT / "scripts" / "verify_repository.py")])

def evaluation_replay():
    verify()
    run([sys.executable, str(ROOT / "scripts" / "replay_stage16.py")])

def patient():
    registry=json.loads((ROOT/"config"/"asset_registry.json").read_text(encoding="utf-8"))
    pending=[x for x in registry["required_for_patient_inference"] if x["status"].startswith("PENDING")]
    if pending:
        ids=", ".join(x["id"] for x in pending)
        raise SystemExit(
            "Patient inference assets are not published yet: "+ids+
            ". Verification/evaluation replay is executable, but the heavyweight "
            "neural inference bundle is still being packaged."
        )
    raise SystemExit("Patient runner wiring is not yet promoted.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["verify","evaluation-replay","patient"],default="verify")
    args=ap.parse_args()
    if args.mode=="verify":
        verify()
    elif args.mode=="evaluation-replay":
        evaluation_replay()
    else:
        patient()

if __name__=="__main__":
    main()
