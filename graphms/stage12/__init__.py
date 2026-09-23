"""Standalone wrapper around the unchanged canonical v2.1 feature engine."""
from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage, stats
from skimage import measure
from graphms.assets import sha256


def engine_namespace(cache_dir, atlas_cache, image_paths, mask_path, case_id, atlas=None):
    """Load function definitions only; do not execute notebook orchestration."""
    source=Path(__file__).with_name('canonical_v2_1_source.py')
    tree=ast.parse(source.read_text())
    namespace=dict(np=np,nib=nib,ndimage=ndimage,stats=stats,measure=measure,
        math=math,os=os,re=re,time=time,Path=Path,json=json,hashlib=hashlib,
        sha256_file=lambda path,chunk=1<<20:sha256(path),
        SEG_THRESHOLD=.5,MIN_COMPONENT_VOXELS=5,CONNECTIVITY=26,
        FEATURE_SCHEMA_VERSION='GraphMSNet-Stage12-MAX-v2.1',
        GLCM_Z_CLIP=5.,GLCM_BIN_WIDTH_Z=.25,GLCM_DISTANCES=(1,2),GLCM_MIN_VOXELS=8,
        MODALITIES=['flair','t1','t2'],EPS=1e-12,
        STRUCT26=ndimage.generate_binary_structure(3,3),
        CACHE_DIR=Path(cache_dir),ATLAS_CACHE=Path(atlas_cache),FORCE_RECOMPUTE=True,
        CHECKPOINT_VERSION='graphms-v3.5.1-stage12-final-hybrid-v1',
        WINNING_RECIPE=Path(__file__).resolve().parents[2]/'precomputed/stage12/FINAL_SEGMENTATION_HANDOFF.json')
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
    namespace['image_triplet']=lambda _case,_role:[Path(p) for p in image_paths]
    namespace['case_from_mask_path']=lambda _path:case_id
    namespace['parse_case_id']=lambda _case:(case_id,'T1')
    if atlas is None:
        # Run the exact canonical Harvard-Oxford loading block, after dependency
        # installation. Never install packages as a side effect of inference.
        import nilearn
        namespace['ensure_nilearn']=lambda:None
        namespace['ENABLE_ATLAS_LOCATION']=True
        block=next(n for n in tree.body if isinstance(n,ast.If) and
                   isinstance(n.test,ast.Name) and n.test.id=='ENABLE_ATLAS_LOCATION')
        exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),namespace)
        if not namespace['ATLAS'].get('enabled'):
            raise RuntimeError('Canonical atlas unavailable. Supply the original --atlas-cache directory. '+
                               str(namespace['ATLAS'].get('reason','')))
    else:
        namespace['ATLAS']=atlas
    namespace['CONFIG_FINGERPRINT']=namespace['feature_config_fingerprint']()
    return namespace


def extract_features(mask_path, image_paths, case_id, output_dir, atlas_cache):
    out=Path(output_dir);cache=out/'feature_cache';cache.mkdir(parents=True,exist_ok=True)
    ns=engine_namespace(cache,atlas_cache,image_paths,mask_path,case_id)
    rec,lesions=ns['extract_case'](Path(mask_path),'research_inference')
    rec['atlas_source']=ns['ATLAS'].get('source')
    rec['atlas_enabled']=bool(ns['ATLAS'].get('enabled'))
    return rec,lesions
