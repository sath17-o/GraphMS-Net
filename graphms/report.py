"""Local, self-contained patient research outputs."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

def save_report(output, case_id, features, lesions, risk, provenance):
    output=Path(output)
    pd.DataFrame([features]).to_csv(output/'features.csv',index=False)
    pd.DataFrame(lesions).to_csv(output/'lesions.csv',index=False)
    (output/'risk.json').write_text(json.dumps(risk,indent=2,allow_nan=False)+'\n')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    rows=[('Lesion volume',f"{features['lesion_volume_ml']:.3f} mL"),
          ('Lesion count',str(features['lesion_count'])),
          ('Predicted EDSS',f"{risk['predicted_edss']:.3f}"),
          ('Estimated probability of EDSS ≥ 4',f"{risk['edss_ge4_probability']:.4f}"),
          ('Fold',str(provenance['fold'])),('Model','GraphMS v3.5.1 Hybrid')]
    table=''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(v)+'</td></tr>' for k,v in rows)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>GraphMS-Net report</title>
<style>body{max-width:900px;margin:40px auto;padding:0 24px;font:17px system-ui;color:#183040}h1{color:#14586b}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}img{max-width:100%}.note{background:#edf4f6;padding:18px}</style>'''
    page+='<h1>GraphMS-Net · '+html.escape(case_id)+'</h1><p class="note">Research output. Development five-fold CV is the validated scope. This selected-fold run does not establish external validation or clinical suitability.</p>'
    page+='<table>'+table+'</table><h2>Lesion overlay</h2><img src="overlay.png" alt="FLAIR MRI with predicted lesions highlighted"><p>Files: lesion_mask.nii.gz · lesion_probability.nii.gz · features.csv · lesions.csv · risk.json · provenance.json</p></html>'
    (output/'patient_report.html').write_text(page,encoding='utf-8')


def save_overlay(flair_path,mask_path,output):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    flair=nib.as_closest_canonical(nib.load(str(flair_path)))
    mask=nib.as_closest_canonical(nib.load(str(mask_path)))
    a=np.asarray(flair.dataobj,np.float32);m=np.asarray(mask.dataobj)>0
    z=int(np.argmax(m.sum(axis=(0,1)))) if m.any() else a.shape[2]//2
    values=a[np.isfinite(a)&(a!=0)]
    low,high=np.percentile(values,[1,99]) if len(values) else (0,1)
    fig,axes=plt.subplots(1,2,figsize=(10,5),facecolor='white')
    for ax in axes:
        ax.imshow(a[:,:,z].T,origin='lower',cmap='gray',vmin=low,vmax=high)
        ax.axis('off')
    overlay=np.ma.masked_where(~m[:,:,z].T,m[:,:,z].T)
    axes[1].imshow(overlay,origin='lower',cmap='autumn',alpha=.65,vmin=0,vmax=1)
    axes[0].set_title('FLAIR');axes[1].set_title('Predicted lesions')
    fig.tight_layout();fig.savefig(Path(output)/'overlay.png',dpi=140);plt.close(fig)
