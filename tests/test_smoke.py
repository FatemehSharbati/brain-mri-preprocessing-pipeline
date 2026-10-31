"""Smoke tests on synthetic data — no downloads needed."""
import numpy as np

from src.io import load_nifti, save_nifti
from src.preprocess import zscore_normalize, resample_isotropic
from src.qc import volume_stats, save_montage


def _phantom(shape=(48, 48, 32), seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ball = ((xx - shape[2] / 2) ** 2 + (yy - shape[1] / 2) ** 2
            + (zz - shape[0] / 2) ** 2) < (min(shape) / 3) ** 2
    return (ball.astype(np.float32) * 100 + rng.normal(0, 5, shape)).astype(np.float32)


def test_normalize():
    data = _phantom()
    normed, mu, sigma = zscore_normalize(data)
    assert abs(normed.mean()) < 1e-4, normed.mean()
    assert abs(normed.std() - 1.0) < 1e-3, normed.std()


def test_resample():
    data = _phantom()
    out, spacing = resample_isotropic(data, input_spacing=(2.0, 2.0, 2.0),
                                      target_spacing=1.0)
    assert out.ndim == 3
    assert out.shape[0] > data.shape[0]  # finer grid -> more voxels


def test_resample_anisotropic_spacing():
    # Physical extent must drive the output grid, not an assumed 1mm spacing:
    # numpy (48, 48, 32) with (2, 2, 3)mm voxels -> physical (64, 96, 144)mm
    # -> 1mm isotropic grid of numpy shape (144, 96, 64).
    data = _phantom()
    out, spacing = resample_isotropic(data, input_spacing=(2.0, 2.0, 3.0),
                                      target_spacing=1.0)
    assert spacing == (1.0, 1.0, 1.0)
    assert out.shape == (144, 96, 64), out.shape


def test_qc(tmp_path):
    data = _phantom()
    stats = volume_stats(data)
    assert stats["snr_estimate"] > 0
    png = save_montage(data, tmp_path / "montage.png", n_slices=4)
    assert png.exists()


def test_io_roundtrip(tmp_path):
    data = _phantom()
    p = save_nifti(data, np.eye(4), tmp_path / "test.nii.gz")
    back, affine, _ = load_nifti(p)
    assert back.shape == data.shape
    assert np.allclose(affine, np.eye(4))
