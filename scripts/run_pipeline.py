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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["verify","evaluation-replay","patient"],default="verify")
    ap.add_argument('--flair', type=Path)
    ap.add_argument('--t1', type=Path)
    ap.add_argument('--t2', type=Path)
    ap.add_argument('--case-id')
    ap.add_argument('--fold', type=int, choices=range(5))
    ap.add_argument('--asset-root', type=Path, default=ROOT/'pretrained')
    ap.add_argument('--atlas-cache', type=Path, default=ROOT/'pretrained'/'atlas_cache')
    ap.add_argument('--output', type=Path)
    args=ap.parse_args()
    if args.mode=='patient':
        missing=[x for x in ['flair','t1','t2','case_id','output'] if getattr(args,x) is None]
        if missing: ap.error('Patient mode requires: '+', '.join('--'+x.replace('_','-') for x in missing))
    if args.mode=="verify":
        verify()
    elif args.mode=="evaluation-replay":
        evaluation_replay()
    else:
        sys.path.insert(0,str(ROOT))
        from graphms.patient import run_patient
        run_patient(args,ROOT)

if __name__=="__main__":
    main()
