# Bundled evaluator demo inputs

This directory contains the three MRI volumes used by the one-click GraphMS
evaluator demonstration for development case `MSLesSeg_P10_T1`:

- `*_0000.nii.gz` — FLAIR
- `*_0001.nii.gz` — T1
- `*_0002.nii.gz` — T2

The larger NIfTI files are stored as byte-exact `.part*` chunks and are
reassembled by `scripts/prepare_demo_inputs.py`. The script verifies the
reconstructed files with committed SHA-256 values before they are used.

No ground-truth lesion mask, EDSS record, or patient metadata is bundled here.

## Source and licence

These demonstration MRI data correspond to the MSLesSeg development case
`MSLesSeg_P10_T1` and are derived from:

Francesco Guarnera, Alessia Rondinella, Elena Crispino *et al.*,
**MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion
Segmentation dataset**, *Scientific Data* 12, 920 (2025).
Article DOI: https://doi.org/10.1038/s41597-025-05250-y
Dataset record: https://doi.org/10.6084/m9.figshare.27919209

MSLesSeg is distributed under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** licence:
https://creativecommons.org/licenses/by/4.0/

The files are included here solely to make the public research reproducibility
demo executable without manual upload. Their inclusion does not change the
GraphMS scientific claim scope: development five-fold CV, not external
validation or clinical deployment.
