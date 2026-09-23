"""Full local research run with explicit fold selection and atomic completion."""
from pathlib import Path
import json
import re
import shutil
import tempfile
from datetime import datetime,timezone
import pandas as pd
from .assets import sha256,PROTOCOL

def resolve_fold(case_id,fold,root):
    table=pd.read_csv(Path(root)/'results/stage16/STAGE16_PER_CASE_METRICS.csv')
    case_col='case' if 'case' in table else 'case_id'
    fold_col='outer' if 'outer' in table else 'fold'
    matches=table[table[case_col].astype(str)==case_id]
    if len(matches):
        known=int(matches.iloc[0][fold_col])
        if fold is not None and fold!=known:
            raise ValueError(f'{case_id} belongs to held-out fold {known}; refusing fold {fold}')
        return known,'development_case_fold_replay'
    if fold is None:
        raise ValueError('For an unseen case, specify --fold 0–4. No validated new-patient ensemble rule exists.')
    return fold,'unseen_case_explicit_fold_research_run'

def run_patient(args,root):
    from .inference import predict_mri
    from .stage12 import extract_features
    from .stage13 import predict_stage13
    from .report import save_report,save_overlay
    case=args.case_id
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',case):
        raise ValueError('Case ID must contain only letters, digits, underscores or hyphens (max 100)')
    fold,scope=resolve_fold(case,args.fold,root)
    paths=[Path(p).expanduser().resolve() for p in (args.flair,args.t1,args.t2)]
    output=Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}. Choose a new output directory.')
    output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        hashes=[sha256(p) for p in paths]
        mask=predict_mri(paths,args.asset_root,fold,work)
        features,lesions=extract_features(mask,paths,case,work,args.atlas_cache)
        risk=predict_stage13(pd.DataFrame([features]),
            Path(root)/'pretrained/stage13/stage13_primary_edss_ge4_classifier.joblib',
            Path(root)/'pretrained/stage13/stage13_primary_edss_regressor.joblib')
        risk['run_scope']=scope
        features['mask_source']='lesion_mask.nii.gz'
        provenance={'status':'COMPLETE','created_at':datetime.now(timezone.utc).isoformat(),
            'case_id':case,'fold':fold,'run_scope':scope,'protocol_sha':PROTOCOL,
            'input_sha256':dict(zip(['FLAIR','T1','T2'],hashes)),
            'asset_lock_sha256':sha256(Path(args.asset_root)/'assets.lock.json'),
            'runtime_source_sha256':{str(p.relative_to(root)):sha256(p) for p in [
                Path(root)/'graphms/inference.py',Path(root)/'graphms/frozen_primitives.py',
                Path(root)/'graphms/stage12/canonical_v2_1_source.py',
                Path(root)/'pretrained/stage13/stage13_primary_edss_ge4_classifier.joblib',
                Path(root)/'pretrained/stage13/stage13_primary_edss_regressor.joblib']},
            'atlas_source':features['atlas_source'],
            'ground_truth_used':False,'new_training_performed':False,
            'validated_claim_scope':'development five-fold CV'}
        save_overlay(paths[0],mask,work)
        save_report(work,case,features,lesions,risk,provenance)
        (work/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','files':{
            p.name:sha256(p) for p in sorted(work.iterdir()) if p.is_file()}},indent=2)+'\n')
        work.rename(output)
    except Exception:
        shutil.rmtree(work,ignore_errors=True)
        raise
    print('GRAPHMS RESEARCH RUN COMPLETE:',output/'patient_report.html')
