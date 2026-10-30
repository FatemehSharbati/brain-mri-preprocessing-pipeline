"""Core preprocessing: bias correction, normalization, resampling."""
import numpy as np
import SimpleITK as sitk


def n4_bias_correction(data, mask=None):
    """N4 bias-field correction via SimpleITK.

    data: 3D numpy array (float). mask: optional 3D binary array.
    Returns corrected 3D float32 array.
    """
    img = sitk.GetImageFromArray(np.asarray(data, dtype=np.float32))
    mask_img = None
    if mask is not None:
        mask_img = sitk.GetImageFromArray((np.asarray(mask) > 0).astype(np.uint8))
        mask_img.CopyInformation(img)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask_img)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def zscore_normalize(data, mask=None):
    """Z-score normalization. Returns (normalized, mean, std)."""
    data = np.asarray(data, dtype=np.float32)
    vals = data if mask is None else data[np.asarray(mask) > 0]
    mu = float(vals.mean())
    sigma = float(vals.std() + 1e-8)
    return ((data - mu) / sigma).astype(np.float32), mu, sigma


def resample_isotropic(data, input_spacing=(1.0, 1.0, 1.0), target_spacing=1.0,
                       interpolator=sitk.sitkLinear):
    """Resample a 3D array to isotropic spacing. Returns (array, new_spacing)."""
    img = sitk.GetImageFromArray(np.asarray(data, dtype=np.float32))
    img.SetSpacing(tuple(float(s) for s in input_spacing))
    in_size = np.array(img.GetSize(), dtype=float)
    in_spacing = np.array(img.GetSpacing(), dtype=float)
    out_spacing = np.full(3, float(target_spacing))
    out_size = np.maximum(1, np.round(in_size * in_spacing / out_spacing)).astype(int)
    resampled = sitk.Resample(
        img,
        [int(s) for s in out_size],
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        tuple(float(s) for s in out_spacing),
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32), tuple(float(s) for s in out_spacing)
