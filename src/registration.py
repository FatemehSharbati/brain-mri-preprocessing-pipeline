"""Rigid/affine registration of a moving volume to a template (SimpleITK)."""
import numpy as np
import SimpleITK as sitk


def register_to_template(moving, fixed, transform_type="affine", shrink_factors=(4, 2, 1),
                         iterations=100):
    """Register `moving` (3D array) to `fixed` (3D array).

    transform_type: "rigid" or "affine". Uses Mattes mutual information.
    Returns (resampled_moving, final_transform).
    """
    moving_img = sitk.GetImageFromArray(np.asarray(moving, dtype=np.float32))
    fixed_img = sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float32))

    initial = sitk.CenteredTransformInitializer(
        fixed_img, moving_img,
        sitk.AffineTransform(fixed_img.GetDimension()) if transform_type == "affine"
        else sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=iterations,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    final_transform = reg.Execute(fixed_img, moving_img)
    resampled = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0,
        moving_img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32), final_transform


def apply_transform(moving, reference, transform):
    """Apply an existing SimpleITK transform of `moving` onto `reference` grid."""
    moving_img = sitk.GetImageFromArray(np.asarray(moving, dtype=np.float32))
    ref_img = sitk.GetImageFromArray(np.asarray(reference, dtype=np.float32))
    out = sitk.Resample(moving_img, ref_img, transform, sitk.sitkLinear, 0.0,
                        moving_img.GetPixelID())
    return sitk.GetArrayFromImage(out).astype(np.float32)
