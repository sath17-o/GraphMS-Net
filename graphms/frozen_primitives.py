"""Inference primitives copied verbatim from the frozen v6 notebook.
No notebook bootstrap, training, Drive adapter or ground truth loading executes.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from acvl_utils.cropping_and_padding.padding import pad_nd_image
CELL_SIZE=4
BRAIN_CELL_MIN_FRACTION=0.001
MAX_NODES_OPERATIONAL=90000
SPATIAL_KNN=8
FEATURE_KNN=4
SEMANTIC_DEEP_GROUPS=16
ENCODER_STAGE_INDICES=(-4,-3,-2,-1)
GAT_HIDDEN=128
PATCH_VOXELS=64
PATCH_CELLS=16
DEVICE=torch.device("cuda")
def pad_to_cell(arr,cell,fill=0):
    shape=arr.shape[-3:]
    target=tuple(int(math.ceil(s/cell)*cell) for s in shape)
    pads=[(0,0)]*(arr.ndim-3)+[(0,target[i]-shape[i]) for i in range(3)]
    return np.pad(arr,pads,mode="constant",constant_values=fill),shape,target

def block_view_3d(arr,cell):
    d,h,w=arr.shape[-3:];gd,gh,gw=d//cell,h//cell,w//cell
    prefix=arr.shape[:-3]
    x=arr.reshape(*prefix,gd,cell,gh,cell,gw,cell)
    axes=list(range(len(prefix)))+[len(prefix),len(prefix)+2,len(prefix)+4,
                                   len(prefix)+1,len(prefix)+3,len(prefix)+5]
    return x.transpose(axes)

def cell_statistics(data,pfg,cell):
    data_p,orig,target=pad_to_cell(data,cell,0)
    p_p,_,_=pad_to_cell(pfg,cell,0)
    brain=np.any(np.abs(data)>1e-6,axis=0)
    brain_p,_,_=pad_to_cell(brain.astype(np.uint8),cell,0)
    pb=block_view_3d(p_p,cell);bb=block_view_3d(brain_p,cell).astype(bool);db=block_view_3d(data_p,cell)
    pmax=pb.max(axis=(-3,-2,-1));pmean=pb.mean(axis=(-3,-2,-1));pstd=pb.std(axis=(-3,-2,-1))
    q=np.clip(pb,1e-6,1-1e-6);ent=-(q*np.log(q)+(1-q)*np.log(1-q))
    modality=[]
    for c in range(data.shape[0]):
        vals=db[c];cnt=bb.sum(axis=(-3,-2,-1));s=(vals*bb).sum(axis=(-3,-2,-1))
        mean=np.divide(s,cnt,out=np.zeros_like(s,dtype=np.float32),where=cnt>0)
        sq=((vals**2)*bb).sum(axis=(-3,-2,-1))
        var=np.divide(sq,cnt,out=np.zeros_like(sq,dtype=np.float32),where=cnt>0)-mean**2
        std=np.sqrt(np.maximum(var,0));vmax=np.where(bb,vals,-np.inf).max(axis=(-3,-2,-1));vmax[~np.isfinite(vmax)]=0
        modality.append((mean.astype(np.float32),std.astype(np.float32),vmax.astype(np.float32)))
    return {
        "orig_shape":orig,"target_shape":target,
        "pmax":pmax.astype(np.float32),"pmean":pmean.astype(np.float32),"pstd":pstd.astype(np.float32),
        "ent_mean":ent.mean(axis=(-3,-2,-1)).astype(np.float32),
        "ent_max":ent.max(axis=(-3,-2,-1)).astype(np.float32),
        "brain_frac":bb.mean(axis=(-3,-2,-1)).astype(np.float32),
        "modality":modality,
    }

def candidate_grid_coords(stats):
    coords=np.argwhere(stats["brain_frac"]>BRAIN_CELL_MIN_FRACTION).astype(np.int32)
    if len(coords)>MAX_NODES_OPERATIONAL:
        raise RuntimeError(
            f"{len(coords)} full-brain nodes exceed operational ceiling {MAX_NODES_OPERATIONAL}; "
            "nodes were NOT truncated."
        )
    assert len(coords)>0
    return coords

def node_bounds_centers(coords,shape):
    bounds=[];centers=[]
    for gz,gy,gx in coords:
        z0,y0,x0=int(gz*CELL_SIZE),int(gy*CELL_SIZE),int(gx*CELL_SIZE)
        z1,y1,x1=min(z0+CELL_SIZE,shape[0]),min(y0+CELL_SIZE,shape[1]),min(x0+CELL_SIZE,shape[2])
        bounds.append((z0,z1,y0,y1,x0,x1))
        centers.append(((z0+z1-1)/2,(y0+y1-1)/2,(x0+x1-1)/2))
    return np.asarray(bounds,np.int32),np.asarray(centers,np.float32)

def node_payload(logit_diff,gt,data,coords):
    K=CELL_SIZE**3
    logits=np.zeros((len(coords),K),np.float32)
    labels=np.zeros((len(coords),K),np.uint8)
    valid=np.zeros((len(coords),K),np.uint8)
    mri=np.zeros((len(coords),3,K),np.float16)
    bounds,centers=node_bounds_centers(coords,gt.shape)
    for i,(z0,z1,y0,y1,x0,x1) in enumerate(bounds):
        l=logit_diff[z0:z1,y0:y1,x0:x1].reshape(-1);g=gt[z0:z1,y0:y1,x0:x1].reshape(-1);n=len(l)
        logits[i,:n]=l;labels[i,:n]=g;valid[i,:n]=1
        for m in range(3):
            mri[i,m,:n]=data[m,z0:z1,y0:y1,x0:x1].reshape(-1).astype(np.float16)
    return logits,labels,valid,mri,bounds,centers

def unwrap_network(net):return net._orig_mod if hasattr(net,"_orig_mod") else net

def sample_feature_map(feat,local_zyx,patch_shape):
    if len(local_zyx)==0:return np.empty((0,int(feat.shape[1])),np.float32)
    ps=np.asarray(patch_shape,np.float32);loc=np.asarray(local_zyx,np.float32);norm=np.empty_like(loc)
    norm[:,0]=2*loc[:,2]/max(ps[2]-1,1)-1
    norm[:,1]=2*loc[:,1]/max(ps[1]-1,1)-1
    norm[:,2]=2*loc[:,0]/max(ps[0]-1,1)-1
    grid=torch.from_numpy(norm).to(feat.device)[None,:,None,None,:]
    sampled=F.grid_sample(feat.float(),grid,mode="bilinear",padding_mode="border",align_corners=True)[0,:,:,0,0].T
    return sampled.detach().cpu().numpy().astype(np.float32)

@torch.inference_mode()
def extract_encoder_nodes(predictor,data_np,centers):
    patch_shape=tuple(map(int,predictor.configuration_manager.patch_size))
    data_t=torch.from_numpy(data_np.astype(np.float32,copy=False))
    padded,revert=pad_nd_image(data_t,patch_shape,"constant",{"value":0},True,None)
    offset=np.asarray([revert[i].start for i in range(1,4)],np.float32)
    centers_pad=centers+offset[None,:]
    slicers=predictor._internal_get_sliding_window_slicers(tuple(padded.shape[1:]))
    net=unwrap_network(predictor.network).to(DEVICE).eval()
    gaussian=compute_gaussian(
        patch_shape,sigma_scale=1/8,value_scaling_factor=10,
        dtype=torch.float32,device=DEVICE
    ).detach().cpu().numpy().astype(np.float32)
    outputs=None;weights=np.zeros(len(centers),np.float32);stage_dims=None
    for sl in slicers:
        start=np.asarray([sl[1].start,sl[2].start,sl[3].start],np.float32)
        end=np.asarray([sl[1].stop,sl[2].stop,sl[3].stop],np.float32)
        ids=np.flatnonzero(np.all((centers_pad>=start[None,:])&(centers_pad<end[None,:]),axis=1))
        if len(ids)==0:continue
        patch=padded[sl][None].to(DEVICE,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16):
            skips=net.encoder(patch)
        selected=[skips[i] for i in ENCODER_STAGE_INDICES]
        if outputs is None:
            stage_dims=[int(v.shape[1]) for v in selected]
            outputs=np.zeros((len(centers),sum(stage_dims)),np.float32)
        local=centers_pad[ids]-start[None,:]
        obs=np.concatenate([sample_feature_map(v,local,patch_shape) for v in selected],axis=1)
        ii=np.rint(local).astype(np.int32)
        for d in range(3):ii[:,d]=np.clip(ii[:,d],0,patch_shape[d]-1)
        w=np.maximum(gaussian[ii[:,0],ii[:,1],ii[:,2]],1e-8)
        outputs[ids]+=obs*w[:,None];weights[ids]+=w
        del patch,skips,selected,obs
    assert outputs is not None and np.all(weights>0)
    outputs/=weights[:,None]
    return outputs,stage_dims

def handcrafted_features(data,pfg,stats,coords):
    gd,gh,gw=stats["pmax"].shape;zz,yy,xx=coords[:,0],coords[:,1],coords[:,2]
    feats=[
        stats["pmean"][zz,yy,xx],stats["pmax"][zz,yy,xx],stats["pstd"][zz,yy,xx],
        stats["ent_mean"][zz,yy,xx],stats["ent_max"][zz,yy,xx],stats["brain_frac"][zz,yy,xx],
    ]
    for mean,std,vmax in stats["modality"]:feats += [mean[zz,yy,xx],std[zz,yy,xx],vmax[zz,yy,xx]]
    grad=np.sqrt(sum(g.astype(np.float32)**2 for g in np.gradient(pfg.astype(np.float32))))
    gp,_,_=pad_to_cell(grad,CELL_SIZE,0);gb=block_view_3d(gp,CELL_SIZE)
    feats += [gb.mean(axis=(-3,-2,-1))[zz,yy,xx],gb.max(axis=(-3,-2,-1))[zz,yy,xx]]
    norm=(coords.astype(np.float32)+.5)/np.maximum(np.array([gd,gh,gw],np.float32),1)[None,:]
    feats += [norm[:,0],norm[:,1],norm[:,2]]
    return np.stack(feats,axis=1).astype(np.float32)

def compact_deep(deep,stage_dims,groups=SEMANTIC_DEEP_GROUPS):
    last=deep[:,-int(stage_dims[-1]):].astype(np.float32)
    return np.stack([last[:,idx].mean(1) for idx in np.array_split(np.arange(last.shape[1]),groups)],axis=1)

def build_edges(coords,hand,deep,stage_dims):
    n=len(coords);desc=np.concatenate([hand[:,:-3],compact_deep(deep,stage_dims)],axis=1).astype(np.float32)
    mu=desc.mean(0,keepdims=True);sd=desc.std(0,keepdims=True);sd[sd<1e-4]=1;desc=(desc-mu)/sd
    flags={}
    def add(i,j,k):
        key=(int(i),int(j))
        if key not in flags:flags[key]=[0,0,0,0]
        flags[key][k]=1
    lookup={tuple(c):i for i,c in enumerate(coords)}
    for i,c in enumerate(coords):
        for dz,dy,dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            j=lookup.get((int(c[0]+dz),int(c[1]+dy),int(c[2]+dx)))
            if j is not None:add(i,j,0)
    if n>1:
        _,idx=cKDTree(coords.astype(np.float32)).query(coords.astype(np.float32),k=min(SPATIAL_KNN+1,n))
        if np.ndim(idx)==1:idx=idx[:,None]
        for i in range(n):
            for j in np.atleast_1d(idx[i])[1:]:add(i,j,1);add(j,i,1)
        _,idx=cKDTree(desc).query(desc,k=min(FEATURE_KNN+1,n))
        if np.ndim(idx)==1:idx=idx[:,None]
        for i in range(n):
            for j in np.atleast_1d(idx[i])[1:]:add(i,j,2);add(j,i,2)
    for i in range(n):add(i,i,3)
    pairs=np.asarray(list(flags),np.int64);fl=np.asarray([flags[tuple(x)] for x in pairs],np.float32)
    src,dst=pairs[:,0],pairs[:,1]
    spatial=np.linalg.norm(coords[src].astype(np.float32)-coords[dst].astype(np.float32),axis=1)
    spatial/=max(float(np.linalg.norm(coords.max(0)-coords.min(0))),1.)
    fdist=np.linalg.norm(desc[src]-desc[dst],axis=1)/max(math.sqrt(desc.shape[1]),1.);fdist=np.clip(fdist,0,5)/5
    unit=desc/np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-8)
    cosine=np.sum(unit[src]*unit[dst],axis=1)
    edge_attr=np.concatenate([spatial[:,None],fdist[:,None],cosine[:,None],fl],axis=1).astype(np.float32)
    return pairs.T.astype(np.int32),edge_attr

def graph_x(a):
    return np.concatenate([
        a["deep"].astype(np.float32),
        a["handcrafted"].astype(np.float32)
    ],1)

def crop_pad(arr,start,size):
    out=np.zeros(arr.shape[:-3]+(size,size,size),dtype=arr.dtype)
    z0,y0,x0=map(int,start);Z,Y,X=arr.shape[-3:]
    zs0=max(z0,0);ys0=max(y0,0);xs0=max(x0,0);zs1=min(z0+size,Z);ys1=min(y0+size,Y);xs1=min(x0+size,X)
    if zs1<=zs0 or ys1<=ys0 or xs1<=xs0:return out
    oz0=zs0-z0;oy0=ys0-y0;ox0=xs0-x0
    out[...,oz0:oz0+zs1-zs0,oy0:oy0+ys1-ys0,ox0:ox0+xs1-xs0]=arr[...,zs0:zs1,ys0:ys1,xs0:xs1]
    return out

def dense_node_patch(a,g,start):
    coords=a["grid_coords"].astype(np.int32);cstart=np.floor(np.asarray(start)/CELL_SIZE).astype(int);rel=coords-cstart[None,:]
    ids=np.flatnonzero(np.all((rel>=0)&(rel<PATCH_CELLS),axis=1));pos=rel[ids]
    stage_dims=list(map(int,a["stage_dims"]));deep=a["deep"].astype(np.float32);scales=[];off=0
    for d in stage_dims:
        grid=np.zeros((d,PATCH_CELLS,PATCH_CELLS,PATCH_CELLS),np.float32)
        if len(ids):grid[:,pos[:,0],pos[:,1],pos[:,2]]=deep[ids,off:off+d].T
        scales.append(grid);off+=d
    hg=np.zeros((GAT_HIDDEN,PATCH_CELLS,PATCH_CELLS,PATCH_CELLS),np.float32);vg=np.zeros_like(hg)
    if len(ids):
        hg[:,pos[:,0],pos[:,1],pos[:,2]]=g["h"].astype(np.float32)[ids].T
        vg[:,pos[:,0],pos[:,1],pos[:,2]]=g["vctx"].astype(np.float32)[ids].T
    return scales,hg,vg,g["global_descriptor"].astype(np.float32)
