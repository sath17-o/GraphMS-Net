# GraphMS-Net — Complete Guide Pipeline Showcase

This document is the evaluator-facing map from the guide pipeline to the frozen implementation. The repository is designed so a user supplies one co-registered patient triplet — **FLAIR, T1 and T2** — and the selected frozen GraphMS pipeline produces the segmentation, lesion analysis, EDSS/risk research outputs, visualization, report and provenance in one run.

## One input → one complete output bundle

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

## Guide-module implementation map

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
| Stage15 training/configuration audit | `results/stage15/` | Research evidence |
| Stage16 final evaluation | `results/stage16/`, `scripts/replay_stage16.py` | Research evaluation evidence |
| Report + visualization | `graphms/report.py` | Patient inference |
| Reproducibility/provenance | `graphms/assets.py`, `provenance.json`, `COMPLETE.json` | Every completed patient run |

Stages 14–16 are development/evaluation modules. They are retained as reproducible evidence but are **not rerun for every patient**. A patient run uses the frozen system selected by that research pipeline.

## One-command patient run

After the original frozen fold assets have been imported with `scripts/setup_assets.py` and the CUDA runtime has passed `scripts/check_neural_runtime.py`:

```sh
python run_graphms.py \
  --case-id MSLesSeg_P10_T1 \
  --flair /path/to/FLAIR.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t2 /path/to/T2.nii.gz \
  --output outputs/MSLesSeg_P10_T1
```

For a known development case, its held-out fold is selected automatically and a conflicting fold is rejected. For a case outside the frozen 93-case registry, `--fold 0..4` must be supplied because no new-patient five-fold ensemble rule was validated.

## Output bundle

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

## Executed end-to-end acceptance evidence

The packaged pipeline was executed on CUDA for development case `MSLesSeg_P10_T1`, held-out fold 0. The independent inference result was compared with the frozen Stage12 reference only after inference completed.

Committed evidence: `evidence/acceptance/MSLesSeg_P10_T1_ACCEPTANCE.json`

- status: **PASS**
- geometry match: **true**
- exact binary mask match: **true**
- mismatched voxels: **0**
- scope: **one-case CUDA acceptance; not all-fold or external validation**

This establishes one-case implementation/replay parity. It does not expand the scientific claim beyond the frozen development five-fold CV result.

## Frozen scientific identity

- System: **GraphMS v3.5.1 Hybrid**
- Protocol SHA: `8944f1a0deef8a7d0eb57118b4b45e3eed0bc803a2c0e93d63d9c9eaba6578a3`
- Stage11: cross-fitted Stage11-v1, 26-connectivity, no morphology
- Development five-fold equal-fold DSC: **0.748049553870**
- Stage13 classifier: **MRI_SPATIAL_SVM, C=30, mri_spatial**
- Stage13 regressor: **MRI_SPATIAL_RIDGE, alpha=30, mri_spatial**
- Multi-task branch: **IMPLEMENTED_EVALUATED_AUXILIARY_NOT_PROMOTED**

The showcase adds no retraining, model reselection, threshold change or scientific substitution.
