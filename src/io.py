"""NIfTI I/O helpers built on nibabel."""
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path):
    """Load a NIfTI file.

    Returns (data float32, affine 4x4, header).
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data, np.asarray(img.affine, dtype=np.float64), img.header


def save_nifti(data, affine, path):
    """Save a float32 NIfTI, creating parent dirs. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))
    return path
