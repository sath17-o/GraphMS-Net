from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphms.assets import import_assets, verify_assets

def main():
    p=argparse.ArgumentParser(description='Import original frozen weights from MSLesSeg_MS or nnunet_v2. No training.')
    p.add_argument('--source-root', type=Path)
    p.add_argument('--asset-root', type=Path, default=Path(__file__).resolve().parents[1]/'pretrained')
    p.add_argument('--folds', type=int, choices=range(5), nargs='+', default=list(range(5)))
    args=p.parse_args()
    if args.source_root:
        import_assets(args.source_root,args.asset_root,args.folds)
    else:
        for fold in args.folds: verify_assets(args.asset_root,fold)
    print('FROZEN ASSETS VERIFIED:', ', '.join(map(str,args.folds)))

if __name__=='__main__': main()
