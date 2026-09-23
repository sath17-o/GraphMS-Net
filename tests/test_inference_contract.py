"""Tests for packaging fidelity and label-free execution contracts."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def v6_tree():
    nb=json.loads((ROOT/'notebooks/frozen_source/GraphMS_v3_5_1_FINAL_FOLDS1_4_IO_OPTIMIZED_v6.ipynb').read_text())
    return ast.parse('\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))

class ContractTests(unittest.TestCase):
    def test_primitives_verbatim(self):
        upstream={n.name:n for n in v6_tree().body if isinstance(n,ast.FunctionDef)}
        extracted=ast.parse((ROOT/'graphms/frozen_primitives.py').read_text())
        count=0
        for n in extracted.body:
            if isinstance(n,ast.FunctionDef):
                self.assertEqual(ast.dump(n),ast.dump(upstream[n.name]),n.name)
                count+=1
        self.assertEqual(count,15)

    def test_fold_cannot_leak(self):
        from graphms.patient import resolve_fold
        self.assertEqual(resolve_fold('MSLesSeg_P10_T1',None,ROOT)[0],0)
        with self.assertRaises(ValueError): resolve_fold('MSLesSeg_P10_T1',1,ROOT)
        with self.assertRaises(ValueError): resolve_fold('NEW_PATIENT',None,ROOT)
        self.assertEqual(resolve_fold('NEW_PATIENT',2,ROOT)[0],2)

    def test_stage11_diagonal_connectivity_and_min_size(self):
        from graphms.postprocessing import apply_stage11
        p=np.zeros((15,15,15),np.float32)
        for i in range(10):p[i,i,i]=.42
        self.assertEqual(int(apply_stage11(p,0).sum()),10)
        self.assertEqual(int(apply_stage11(p,2).sum()),0)
        p[9,9,9]=0
        self.assertEqual(int(apply_stage11(p,0).sum()),0)

    def test_asset_tamper_detection(self):
        from graphms.assets import asset_paths,sha256,verify_assets,PROTOCOL
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files={}
            for rel in asset_paths(0):
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                files[rel]={'sha256':sha256(p),'size_bytes':7}
            (root/'assets.lock.json').write_text(json.dumps({'protocol_sha':PROTOCOL,'files':files}))
            verify_assets(root,0)
            (root/asset_paths(0)[-1]).write_bytes(b'changed')
            with self.assertRaises(RuntimeError):verify_assets(root,0)

    def test_geometry_rejects_misaligned_modality(self):
        import SimpleITK as sitk
        from graphms.inference import validate_images
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i in range(3):
                im=sitk.GetImageFromArray(np.ones((5,6,7),np.float32))
                if i==2: im.SetOrigin((1.,0.,0.))
                p=Path(d)/f'{i}.nii.gz';sitk.WriteImage(im,str(p));paths.append(p)
            with self.assertRaises(ValueError):validate_images(paths)

    def test_canonical_features_feed_actual_stage13(self):
        import nibabel as nib
        import pandas as pd
        from graphms.stage12 import engine_namespace
        from graphms.stage13 import predict_stage13
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';cache.mkdir()
            grid=np.indices((12,12,12)).sum(0).astype(np.float32)+1
            affine=np.eye(4);affine[:3,3]=[-6,-6,-6]
            paths=[]
            for i in range(3):
                p=root/f'mri{i}.nii.gz';nib.save(nib.Nifti1Image(grid+i,affine),p);paths.append(p)
            mask=np.zeros(grid.shape,np.uint8);mask[3:7,3:7,3:7]=1
            mp=root/'mask.nii.gz';nib.save(nib.Nifti1Image(mask,affine),mp)
            ns=engine_namespace(cache,root/'atlas',paths,mp,'synthetic',atlas={'enabled':False,'reason':'unit-test fixture'})
            features,lesions=ns['extract_case'](mp,'research_inference')
            self.assertEqual(features['lesion_voxels'],64)
            self.assertEqual(features['lesion_count'],1)
            self.assertFalse(features['ground_truth_mask_used'])
            risk=predict_stage13(pd.DataFrame([features]),ROOT/'pretrained/stage13/stage13_primary_edss_ge4_classifier.joblib',ROOT/'pretrained/stage13/stage13_primary_edss_regressor.joblib')
            self.assertTrue(0<=risk['predicted_edss']<=10)
            self.assertTrue(0<=risk['edss_ge4_probability']<=1)

    def test_model_forward_matches_original(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from graphms.models.gat import TrueGAT
        from graphms.models.hybrid import SpatialHybrid
        names={'segment_softmax','SparseEdgeGATLayer','TrueGAT','SE3D','BottleneckSelfAttention','ConvBlock','GuideSpatialHybrid'}
        ns=dict(torch=torch,nn=nn,F=F,CELL_SIZE=4,GAT_HIDDEN=128,GAT_HEADS=4,GUIDE_DROPOUT=.3,FPN_CHANNELS=32)
        selected=[n for n in v6_tree().body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=selected,type_ignores=[]),'<frozen-v6>','exec'),ns)
        torch.set_num_threads(2);torch.manual_seed(4)
        actual=TrueGAT(24).eval();reference=ns['TrueGAT'](24).eval()
        reference.load_state_dict(actual.state_dict())
        args=(torch.randn(4,24),torch.tensor([[0,1,2,3,0,1],[0,1,2,3,1,0]]),torch.randn(6,7),torch.randn(4,64),torch.randn(4,3,64))
        with torch.inference_mode():
            for a,b in zip(actual(*args),reference(*args)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        dims=[8,16,24,32]
        actual=SpatialHybrid(dims).eval();reference=ns['GuideSpatialHybrid'](dims).eval()
        # Avoid an identity output head concealing hidden-layer mismatches.
        with torch.no_grad():actual.out.weight.normal_(0,.1)
        reference.load_state_dict(actual.state_dict())
        args=([torch.randn(1,c,16,16,16) for c in dims],torch.randn(1,128,16,16,16),torch.randn(1,128,16,16,16),torch.randn(1,256),torch.randn(1,1,64,64,64))
        with torch.inference_mode():torch.testing.assert_close(actual(*args),reference(*args),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
