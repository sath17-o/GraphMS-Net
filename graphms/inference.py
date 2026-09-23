"""Label-free execution of the frozen fold-specific CNN → GAT → Hybrid path."""
from __future__ import annotations
import gc
import itertools
import os
from pathlib import Path
import numpy as np
import torch
from .assets import verify_assets, validate_checkpoint
from .models.gat import TrueGAT
from .models.hybrid import SpatialHybrid
from .postprocessing import apply_stage11


def require_cuda():
    from packaging.version import Version
    from importlib.metadata import version
    if version('nnunetv2') != '2.8.1':
        raise RuntimeError('Frozen runtime requires nnunetv2==2.8.1')
    tv=Version(torch.__version__.split('+')[0])
    if tv < Version('2.1.2') or (tv.major==2 and tv.minor==9):
        raise RuntimeError('Require PyTorch >=2.1.2, excluding 2.9.x')
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen neural inference requires CUDA. Run on the original Colab/Kaggle GPU runtime.')


def validate_images(paths):
    import SimpleITK as sitk
    if len(paths) != 3:
        raise ValueError('Exactly FLAIR, T1 and T2 are required, in that order')
    reference = None
    for name, path in zip(('FLAIR','T1','T2'), paths):
        image = sitk.ReadImage(str(path))
        if image.GetDimension()!=3 or image.GetNumberOfComponentsPerPixel()!=1:
            raise ValueError(f'{name}: expected a scalar 3-D NIfTI')
        if min(image.GetSize()) < 2 or min(image.GetSpacing()) <= 0:
            raise ValueError(f'{name}: invalid geometry')
        data=sitk.GetArrayFromImage(image)
        if not np.isfinite(data).all() or not np.any(np.abs(data)>0):
            raise ValueError(f'{name}: empty or nonfinite MRI')
        if reference is not None:
            if image.GetSize()!=reference.GetSize() or any(not np.allclose(a,b,rtol=0,atol=1e-3) for a,b in [
                (image.GetSpacing(),reference.GetSpacing()),(image.GetOrigin(),reference.GetOrigin()),
                (image.GetDirection(),reference.GetDirection())]):
                raise ValueError(f'{name}: MRI modalities must already share the FLAIR geometry')
        else:
            reference=image
    return reference


def patch_starts(shape, size=64, stride=32):
    axes=[]
    for extent in shape:
        last=max(0,int(extent)-size)
        starts=list(range(0,last+1,stride)) or [0]
        if starts[-1]!=last: starts.append(last)
        axes.append(starts)
    return itertools.product(*axes)


@torch.inference_mode()
def infer_gat(features, checkpoint, device):
    from .frozen_primitives import graph_x
    mu=np.asarray(checkpoint['mu'],np.float32)
    sd=np.asarray(checkpoint['sd'],np.float32)
    raw=graph_x(features)
    if mu.shape!=(raw.shape[1],) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all():
        raise ValueError('GAT normalization shape/values mismatch')
    model=TrueGAT(raw.shape[1]).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    def tensor(a): return torch.from_numpy(a).to(device)
    logits,h,vctx,gd=model(
        tensor((raw-mu[None,:])/np.maximum(sd[None,:],1e-6)),
        tensor(features['edge_index'].astype(np.int64)),
        tensor(features['edge_attr'].astype(np.float32)),
        tensor(features['node_voxel_logits'].astype(np.float32)),
        tensor(features['node_voxel_mri'].astype(np.float32)))
    # Preserve the exact cache round-trip precision used by frozen v6 inference.
    result={k:v.float().cpu().numpy().astype(np.float16) for k,v in
            [('h',h),('vctx',vctx),('global_descriptor',gd)]}
    del model,logits,h,vctx,gd
    return result


@torch.inference_mode()
def infer_hybrid(features, gat_cache, checkpoint, device):
    from .frozen_primitives import dense_node_patch,crop_pad
    dims=list(map(int,features['stage_dims']))
    if dims!=list(map(int,checkpoint['stage_dims'])):
        raise ValueError('CNN encoder dimensions do not match Hybrid checkpoint')
    model=SpatialHybrid(dims).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    base=features['base_logits_2ch'].astype(np.float32)
    diff=(base[1]-base[0]).astype(np.float32)
    shape=diff.shape
    target=tuple(max(n,64) for n in shape)
    accum=np.zeros(target,np.float32);count=np.zeros(target,np.float32)
    for start in patch_starts(shape):
        scales,h,v,gd=dense_node_patch(features,gat_cache,start)
        ctx=crop_pad(diff[None],start,64)
        def tensor(a): return torch.from_numpy(a[None]).to(device)
        with torch.autocast('cuda',dtype=torch.float16):
            out=model([tensor(a) for a in scales],tensor(h),tensor(v),tensor(gd),tensor(ctx))[0,0]
        out=out.float().cpu().numpy()
        slices=tuple(slice(s,s+64) for s in start)
        accum[slices]+=out;count[slices]+=1
    if not np.all(count>0):
        raise RuntimeError('Hybrid window coverage failure')
    result=(accum/np.maximum(count,1e-6))[tuple(slice(0,n) for n in shape)]
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite Hybrid logits')
    del model
    return result


def predict_mri(paths, asset_root, fold, output_dir):
    require_cuda()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    import SimpleITK as sitk
    from . import frozen_primitives as fp
    paths=[Path(p).resolve() for p in paths]
    reference=validate_images(paths)
    asset_root=Path(asset_root).resolve()
    verify_assets(asset_root,fold)
    # Validate original checkpoint identities before constructing any network.
    cnn_ck=validate_checkpoint(asset_root/f'resencm250/fold_{fold}/checkpoint_final.pth','resencm250')
    del cnn_ck
    gat_ck=validate_checkpoint(asset_root/f'gat/fold_{fold}/checkpoint_final.pth','gat')
    hybrid_ck=validate_checkpoint(asset_root/f'fusion/fold_{fold}/checkpoint_final.pth','fusion')
    os.environ['nnUNet_compile']='false'
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda')
    predictor=nnUNetPredictor(tile_step_size=0.5,use_gaussian=True,use_mirroring=True,
        perform_everything_on_device=True,device=device,verbose=False,
        verbose_preprocessing=False,allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(asset_root/'resencm250'),
        use_folds=(fold,),checkpoint_name='checkpoint_final.pth')
    print('Preprocessing FLAIR/T1/T2 with frozen nnU-Net plan',flush=True)
    pre=predictor.configuration_manager.preprocessor_class(verbose=False)
    data,_,props=pre.run_case([str(p) for p in paths],None,predictor.plans_manager,
        predictor.configuration_manager,predictor.dataset_json)
    with torch.inference_mode():
        logits=predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data.astype(np.float32,copy=False))).float().cpu()
    base=logits.numpy().astype(np.float32)
    pfg=torch.softmax(logits,dim=0)[1].numpy().astype(np.float32)
    print('Building frozen Stage5/6 graph features',flush=True)
    stats=fp.cell_statistics(data,pfg,fp.CELL_SIZE)
    coords=fp.candidate_grid_coords(stats)
    # node_payload needs an array for shape/unused labels. This zero array is
    # never ground truth and is never consumed by inference.
    dummy=np.zeros(data.shape[1:],np.uint8)
    vl,_,_,vm,bounds,centers=fp.node_payload(base[1]-base[0],dummy,data,coords)
    deep,dims=fp.extract_encoder_nodes(predictor,data,centers)
    hand=fp.handcrafted_features(data,pfg,stats,coords)
    edges,edge_attr=fp.build_edges(coords,hand,deep,dims)
    features={'deep':deep.astype(np.float16),'handcrafted':hand.astype(np.float32),
        'grid_coords':coords.astype(np.int16),'bounds':bounds.astype(np.int16),
        'edge_index':edges.astype(np.int32),'edge_attr':edge_attr.astype(np.float32),
        'node_voxel_logits':vl.astype(np.float32),'node_voxel_mri':vm.astype(np.float16),
        'base_logits_2ch':base,'stage_dims':np.asarray(dims,np.int16)}
    # Free the CNN before the downstream graph/fusion networks.
    predictor.network.to('cpu')
    del data,deep,hand,logits,pfg,dummy,pre
    torch.cuda.empty_cache();gc.collect()
    print('Executing frozen GAT',flush=True)
    g=infer_gat(features,gat_ck,device)
    del gat_ck
    torch.cuda.empty_cache()
    print('Executing frozen Hybrid windows (64 voxels, stride 32)',flush=True)
    diff=infer_hybrid(features,g,hybrid_ck,device)
    del g,hybrid_ck,features
    # v6 make_twoch preserves the original background logit exactly.
    twoch=np.stack([base[0],base[0]+diff],axis=0).astype(np.float32)
    lm=predictor.plans_manager.get_label_manager(predictor.dataset_json)
    _,probabilities=convert_predicted_logits_to_segmentation_with_correct_shape(
        torch.from_numpy(twoch),predictor.plans_manager,predictor.configuration_manager,lm,props,
        return_probabilities=True,num_threads_torch=2)
    probability=np.asarray(probabilities[1],np.float32)
    expected=tuple(reversed(reference.GetSize()))
    if probability.shape!=expected or not np.isfinite(probability).all():
        raise RuntimeError('Output probability geometry/values invalid')
    mask=apply_stage11(probability,fold)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for name,array in [('lesion_probability.nii.gz',probability),('lesion_mask.nii.gz',mask)]:
        image=sitk.GetImageFromArray(array);image.CopyInformation(reference)
        sitk.WriteImage(image,str(output/name))
    return output/'lesion_mask.nii.gz'
