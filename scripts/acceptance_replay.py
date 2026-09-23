"""Run one held-out case end to end, then compare its saved frozen mask."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graphms.patient import resolve_fold
from graphms.assets import import_assets,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--case-id',default='MSLesSeg_P10_T1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,default=ROOT/'pretrained')
    p.add_argument('--atlas-cache',type=Path,default=ROOT/'pretrained/atlas_cache')
    args=p.parse_args()
    fold,scope=resolve_fold(args.case_id,None,ROOT)
    if scope!='development_case_fold_replay':
        p.error('Acceptance replay requires a registered development case')
    nnroot=args.project_root/'nnunet_v2' if (args.project_root/'nnunet_v2').is_dir() else args.project_root
    import_assets(nnroot,args.asset_root,[fold])
    images=nnroot/'nnUNet_raw/Dataset001_MSLesSeg/imagesTr'
    cmd=[sys.executable,str(ROOT/'scripts/run_pipeline.py'),'--mode','patient',
         '--case-id',args.case_id,'--asset-root',str(args.asset_root),
         '--atlas-cache',str(args.atlas_cache),'--output',str(args.output)]
    for i,key in enumerate(['--flair','--t1','--t2']):
        cmd += [key,str(images/f'{args.case_id}_{i:04d}.nii.gz')]
    subprocess.run(cmd,check=True)
    # Reference predictions enter only AFTER the independent inference process.
    import SimpleITK as sitk
    predicted=args.output/'lesion_mask.nii.gz'
    reference=nnroot/'graphms_hybrid_stage12_final_v1/final_hybrid_masks'/f'{args.case_id}.nii.gz'
    pred_img=sitk.ReadImage(str(predicted));ref_img=sitk.ReadImage(str(reference))
    a=sitk.GetArrayFromImage(pred_img);b=sitk.GetArrayFromImage(ref_img)
    geometry=(pred_img.GetSize()==ref_img.GetSize() and all(np.allclose(x,y,atol=1e-3,rtol=0) for x,y in [
        (pred_img.GetOrigin(),ref_img.GetOrigin()),(pred_img.GetSpacing(),ref_img.GetSpacing()),
        (pred_img.GetDirection(),ref_img.GetDirection())]))
    equal=geometry and np.array_equal(a>0,b>0)
    result={'status':'PASS' if equal else 'FAIL','case_id':args.case_id,'fold':fold,
            'comparison':'exact binary mask equality plus geometry',
            'geometry_matches':bool(geometry),'mask_matches':bool(equal),
            'prediction_sha256':sha256(predicted),'reference_sha256':sha256(reference),
            'mismatched_voxels':int(np.count_nonzero((a>0)!=(b>0))) if a.shape==b.shape else None,
            'scope':'one-case CUDA acceptance; not all-fold or external validation'}
    (args.output/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not equal:raise SystemExit('Frozen-mask replay mismatch. Do not promote as exact replay.')

if __name__=='__main__':main()
