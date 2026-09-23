"""Import original frozen artifacts from a user-selected local project tree."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

PROTOCOL = '8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3'
GAT_IMPL = 'v3.4.1-gat-fp32-dice-bce-adamw-cosine-baseline-preserving-selection'
FUSION_IMPL = 'v3.5-concat-se-self-multiscale-dice-bce-adamw-cosine-250iter'
EXPERIMENT = 'graphms_resencm250_true_hybrid_v3_5_1_8944f1a0de'
CNN_DIR = 'nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def asset_paths(fold):
    if fold not in range(5):
        raise ValueError('Fold must be 0–4')
    return [f'resencm250/{n}' for n in ('plans.json', 'dataset.json')] + [
        f'{kind}/fold_{fold}/checkpoint_final.pth' for kind in ('resencm250', 'gat', 'fusion')]

def verify_assets(asset_root, fold):
    root = Path(asset_root)
    lock_path = root / 'assets.lock.json'
    if not lock_path.is_file():
        raise RuntimeError('Missing assets.lock.json. Run scripts/setup_assets.py --source-root PATH --folds '+str(fold))
    lock = json.loads(lock_path.read_text())
    if lock.get('protocol_sha') != PROTOCOL:
        raise RuntimeError('Asset lock protocol mismatch')
    for rel in asset_paths(fold):
        record = lock.get('files', {}).get(rel)
        path = root / rel
        if not record or not path.is_file():
            raise RuntimeError(f'Missing locked asset: {rel}')
        if path.stat().st_size != record['size_bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Asset integrity mismatch: {rel}')
    return lock

def validate_checkpoint(path, kind):
    # Only call on originals explicitly selected through --source-root, or on
    # files whose digest matches the local lock. PyTorch checkpoints use pickle.
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f'{kind}: invalid checkpoint')
    if kind == 'resencm250':
        if not {'network_weights', 'init_args', 'trainer_name'} <= ck.keys():
            raise ValueError('Invalid nnU-Net checkpoint schema')
        if ck['trainer_name'] != 'nnUNetTrainer_250epochs':
            raise ValueError('Wrong CNN trainer')
        state = ck['network_weights']
    else:
        if ck.get('protocol_sha256') != PROTOCOL:
            raise ValueError(f'{kind}: wrong protocol')
        expected = GAT_IMPL if kind == 'gat' else FUSION_IMPL
        if ck.get('implementation_id') != expected:
            raise ValueError(f'{kind}: wrong implementation')
        if kind == 'gat' and not {'mu','sd','model'} <= ck.keys():
            raise ValueError('GAT normalization missing')
        if kind == 'fusion' and (ck.get('gat_phase') != 'refit' or len(ck.get('stage_dims', [])) != 4):
            raise ValueError('Fusion is not the four-scale refit model')
        state = ck['model']
    if not state or any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
        raise ValueError(f'{kind}: empty or nonfinite weights')
    return ck

def import_assets(source_root, asset_root, folds):
    source = Path(source_root).expanduser().resolve()
    nnroot = source / 'nnunet_v2' if (source / 'nnunet_v2').is_dir() else source
    cnn = nnroot / 'nnUNet_results' / 'Dataset001_MSLesSeg' / CNN_DIR
    downstream = nnroot / EXPERIMENT / 'outer_folds'
    dest = Path(asset_root).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock_path = dest / 'assets.lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {
        'schema_version': 1, 'protocol_sha': PROTOCOL,
        'trust': 'locally attested from explicitly selected original project; not a publisher-signed release',
        'files': {}}
    if lock.get('protocol_sha') != PROTOCOL:
        raise ValueError('Existing asset lock has a different protocol')
    pairs = [(cnn / n, f'resencm250/{n}', None) for n in ('plans.json','dataset.json')]
    for fold in sorted(set(folds)):
        asset_paths(fold)
        pairs.append((cnn / f'fold_{fold}/checkpoint_final.pth', f'resencm250/fold_{fold}/checkpoint_final.pth', 'resencm250'))
        for kind in ('gat','fusion'):
            pairs.append((downstream / f'outer_{fold}/{kind}/refit/checkpoint_final.pth', f'{kind}/fold_{fold}/checkpoint_final.pth', kind))
    missing = [str(p) for p, _, _ in pairs if not p.is_file()]
    if missing:
        raise FileNotFoundError('Original artifacts missing:\n'+'\n'.join(missing))
    dataset = json.loads((cnn / 'dataset.json').read_text())
    channels = dataset.get('channel_names', {})
    if [str(channels.get(str(i), '')).lower() for i in range(3)] != ['flair','t1','t2']:
        raise ValueError('Expected FLAIR/T1/T2 channel order')
    plans = json.loads((cnn / 'plans.json').read_text())
    if plans.get('plans_name') != 'nnUNetResEncUNetMPlans' or '3d_fullres' not in plans.get('configurations', {}):
        raise ValueError('Wrong ResEncM plan')
    for src, rel, kind in pairs:
        before = sha256(src)
        if kind:
            ck = validate_checkpoint(src, kind)
            del ck
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src != target and (not target.exists() or sha256(target) != before):
            tmp = target.with_suffix(target.suffix + '.partial')
            try:
                shutil.copyfile(src, tmp)
                if sha256(tmp) != before:
                    raise RuntimeError(f'Copy integrity mismatch: {rel}')
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
        lock['files'][rel] = {'sha256': before, 'size_bytes': target.stat().st_size}
        print('Imported', rel, flush=True)
    tmp = lock_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(lock, indent=2)+'\n')
    tmp.replace(lock_path)
    for fold in folds:
        verify_assets(dest, fold)
    return lock
