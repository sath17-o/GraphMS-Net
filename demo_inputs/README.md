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

These demonstration MRI data are derived from:

Ali M. Muslim, **Brain MRI Dataset of Multiple Sclerosis with Consensus Manual
Lesion Segmentation and Patient Meta Information**, Mendeley Data, Version 1,
2022. DOI: https://doi.org/10.17632/8bctsm8jz7.1

The source dataset is published under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** licence:
https://creativecommons.org/licenses/by/4.0/

The files are included here solely to make the public research reproducibility
demo executable without manual upload. Their inclusion does not change the
GraphMS scientific claim scope: development five-fold CV, not external
validation or clinical deployment.
