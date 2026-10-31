# Brain MRI Preprocessing Pipeline

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A clean, reproducible preprocessing pipeline for T1-weighted brain MRI:
**N4 bias-field correction → intensity normalization → resampling → QC report**,
with NIfTI I/O and an optional rigid/affine registration step to a template.

Built as a portfolio project for MSc/PhD applications in neuroimaging.

## Pipeline

```
input.nii.gz
   │  N4 bias field correction (SimpleITK)
   ▼
   │  z-score intensity normalization
   ▼
   │  isotropic resampling (SimpleITK)
   ▼
   │  (optional) rigid/affine registration to template
   ▼
output.nii.gz  +  qc_report.html
```

## Quickstart

```bash
pip install -r requirements.txt

python -m src.pipeline \
    --input data/sub-01_T1w.nii.gz \
    --output-dir out/sub-01 \
    --spacing 1.0

# skip bias correction, keep everything else
python -m src.pipeline --input data/sub-01_T1w.nii.gz --output-dir out/sub-01 --no-bias

# with registration to a template (e.g. MNI152)
python -m src.pipeline --input data/sub-01_T1w.nii.gz --output-dir out/sub-01 \
    --template templates/MNI152_T1_1mm.nii.gz
```

Outputs in `--output-dir`:
- `preprocessed.nii.gz` — the processed volume
- `qc_report.html` — slice montage + basic stats (mean, std, SNR estimate)
- `qc_montage.png`

## Modules

| Module | What it does |
|---|---|
| `src/io.py` | NIfTI load/save helpers (nibabel) |
| `src/preprocess.py` | N4 bias correction, z-score normalization, isotropic resampling |
| `src/registration.py` | Rigid/affine registration to a template (SimpleITK, Mattes MI) |
| `src/qc.py` | Slice montage + stats → PNG and self-contained HTML report |
| `src/pipeline.py` | CLI that chains everything |

## Notes & scope

- Skull-stripping is intentionally left to dedicated tools (HD-BET, SynthSeg) —
  the pipeline accepts already skull-stripped input and documents it.
- Resampling uses the true voxel spacing read from the NIfTI header
  (anisotropic clinical scans resample to the correct physical extent).
- Tested on CPU with synthetic data (`tests/test_smoke.py`); public test data:
  [IXI](https://brain-development.org/ixi-dataset/) and
  [OASIS](https://www.oasis-brains.org/).
- Not a clinical tool — research/educational use.

## License

MIT — see `LICENSE`.
