# GraphMS-Net — Pipeline Architecture and Reproducibility Map

This document maps the frozen GraphMS research pipeline to its repository implementation and committed evidence. A co-registered **FLAIR, T1 and T2** MRI triplet is propagated through the selected frozen inference path to generate lesion segmentation, lesion-level characterization, downstream MRI-derived EDSS/risk research outputs, visualization, reporting, and provenance.

## End-to-end frozen inference path

```text
FLAIR.nii.gz + T1.nii.gz + T2.nii.gz
                 |
                 v
Input / geometry validation
                 |
                 v
Frozen nnU-Net preprocessing + ResEncM-250 CNN
                 |
                 v
Stage5/6 graph feature construction
                 |
                 v
Stage7 TrueGAT
                 |
                 v
CNN + GNN concatenation
                 |
                 v
GraphMS v3.5.1 Hybrid
(SE + self-attention + multi-scale fusion + decoder + CNN-logit skip)
                 |
                 v
1x1x1 lesion head / probability
                 |
                 v
Cross-fitted Stage11-v1
                 |
                 v
Final lesion mask in original FLAIR geometry
                 |
                 v
Canonical Stage12 v2.1 lesion / MRI-spatial features
                 |
                 v
Stage13 MRI_SPATIAL_SVM + MRI_SPATIAL_RIDGE
                 |
                 v
overlay.png + CSV + JSON + NIfTI + patient_report.html + provenance
```

## Module-to-implementation correspondence

| Pipeline module | Frozen implementation / evidence | Runtime role |
|---|---|---|
| Multimodal MRI input | `graphms/patient.py`, `graphms/inference.py` | Patient inference |
| Input and geometry validation | `graphms/inference.py` | Patient inference |
| CNN / ResEncM-250 | imported frozen nnU-Net fold checkpoint; registry in `config/asset_registry.json` | Patient inference |
| Stage5/6 graph features | `graphms/frozen_primitives.py` | Patient inference |
| Stage7 TrueGAT | `graphms/models/gat.py` + frozen fold checkpoint | Patient inference |
| Stage8/9 Hybrid fusion | `graphms/models/hybrid.py` + frozen fold checkpoint | Patient inference |
| Lesion probability/head | `graphms/inference.py` | Patient inference |
| Stage11 post-processing | `graphms/postprocessing.py` | Patient inference |
| Stage12 lesion analysis | `graphms/stage12/canonical_v2_1_source.py` | Patient inference |
| Stage13 EDSS/risk | `graphms/stage13.py` + committed SVM/Ridge artifacts | Patient inference |
| Stage14 multi-task branch | `results/stage14/` | Research evidence; evaluated, auxiliary, not promoted |
| Stage15 training/configuration verification | `results/stage15/` | Research evidence |
| Stage16 final evaluation | `results/stage16/`, `scripts/replay_stage16.py` | Research evaluation evidence |
| Report + visualization | `graphms/report.py` | Patient inference |
| Reproducibility/provenance | `graphms/assets.py`, `provenance.json`, `COMPLETE.json` | Every completed patient run |

Stages 14–16 are development/evaluation modules. They are retained as reproducible evidence but are **not rerun for every patient**. A patient run uses the frozen system selected by that research pipeline.

## Patient-level inference entry point

After the CUDA runtime has passed `scripts/check_neural_runtime.py`, the
research user supplies the three MRI volumes and executes the frozen inference entry point. If the required
fold is not already present, the runner downloads the frozen assets from the
`assets-v1` GitHub Release and verifies their SHA-256 identities before
starting inference:

```sh
python run_graphms.py \
  --case-id MSLesSeg_P10_T1 \
  --flair /path/to/FLAIR.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t2 /path/to/T2.nii.gz \
  --output outputs/MSLesSeg_P10_T1
```

For a known development case, its held-out fold is selected automatically and a conflicting fold is rejected. For a case outside the frozen 93-case registry, `--fold 0..4` must be supplied because no new-patient five-fold ensemble rule was validated.

The original Drive workspace is an owner-side publication/recovery source, not
a runtime dependency for reproducibility execution. The public `assets-v1` release has been
published and a fresh-clone execution without the original Drive model workspace completed
successfully.

## Generated research outputs

A successful run creates:

```text
outputs/<case-id>/
├── lesion_probability.nii.gz
├── lesion_mask.nii.gz
├── features.csv
├── lesions.csv
├── risk.json
├── overlay.png
├── patient_report.html
├── provenance.json
└── COMPLETE.json
```

The output directory is written atomically. Failed runs do not leave a false completion record.

## End-to-end reproducibility verification

The packaged pipeline was executed on CUDA for development case `MSLesSeg_P10_T1`, held-out fold 0. The independent inference result was compared with the frozen Stage12 reference only after inference completed.

Committed verification record: `evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json`

- status: **PASS**
- geometry match: **true**
- exact binary mask match: **true**
- mismatched voxels: **0**
- scope: **one-case CUDA reproducibility verification; not all-fold or external validation**

This establishes one-case implementation/replay parity. In addition,
`evidence/acceptance/NO_DRIVE_EVALUATOR_ACCEPTANCE.json` records a successful
fresh-clone reproducibility execution using the public release assets rather
than the original Drive model workspace. Neither verification record expands the
scientific claim beyond the finalized development five-fold CV result.

## Frozen scientific identity

- System: **GraphMS v3.5.1 Hybrid**
- Protocol SHA: `8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3`
- Stage11: cross-fitted Stage11-v1, 26-connectivity, no morphology
- Development five-fold equal-fold DSC: **0.748049553870**
- Stage13 classifier: **MRI_SPATIAL_SVM, C=30, mri_spatial**
- Stage13 regressor: **MRI_SPATIAL_RIDGE, alpha=30, mri_spatial**
- Multi-task branch: **IMPLEMENTED_EVALUATED_AUXILIARY_NOT_PROMOTED**

The repository packaging adds no retraining, model reselection, threshold change, or scientific substitution.
