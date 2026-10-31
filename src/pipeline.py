"""CLI: chain bias correction -> normalization -> resampling -> QC report."""
import argparse
from pathlib import Path

from .io import load_nifti, save_nifti
from .preprocess import n4_bias_correction, zscore_normalize, resample_isotropic
from .registration import register_to_template
from .qc import volume_stats, save_montage, write_html_report


def main():
    ap = argparse.ArgumentParser(description="Brain MRI preprocessing pipeline")
    ap.add_argument("--input", required=True, help="input NIfTI (.nii/.nii.gz)")
    ap.add_argument("--output-dir", required=True, help="output directory")
    ap.add_argument("--no-bias", action="store_true", help="skip N4 bias correction")
    ap.add_argument("--no-norm", action="store_true", help="skip z-score normalization")
    ap.add_argument("--spacing", type=float, default=1.0, help="target isotropic spacing")
    ap.add_argument("--template", default=None, help="optional template NIfTI for registration")
    ap.add_argument("--transform", default="affine", choices=["rigid", "affine"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, affine, header = load_nifti(args.input)
    print(f"loaded {args.input} shape={data.shape}")

    if not args.no_bias:
        print("N4 bias correction ...")
        data = n4_bias_correction(data)

    if not args.no_norm:
        print("z-score normalization ...")
        data, mu, sigma = zscore_normalize(data)
        print(f"  mean={mu:.3f} std={sigma:.3f}")

    # Use the true voxel spacing from the header — assuming 1mm would distort
    # scans with anisotropic voxels (e.g. 0.9 x 0.9 x 2.0 mm clinical T1w).
    input_spacing = tuple(float(z) for z in header.get_zooms()[:3])
    print(f"resampling from {input_spacing}mm to {args.spacing}mm isotropic ...")
    data, _ = resample_isotropic(data, input_spacing=input_spacing,
                                 target_spacing=args.spacing)

    if args.template:
        print(f"registering to template {args.template} ({args.transform}) ...")
        fixed, _, _ = load_nifti(args.template)
        data, _ = register_to_template(data, fixed, transform_type=args.transform)

    out_nii = out_dir / "preprocessed.nii.gz"
    save_nifti(data, affine, out_nii)
    print(f"wrote {out_nii}")

    stats = volume_stats(data)
    montage = save_montage(data, out_dir / "qc_montage.png")
    write_html_report(stats, montage, out_dir / "qc_report.html",
                      title=f"QC Report — {Path(args.input).name}")
    print(f"wrote {out_dir / 'qc_report.html'}")
    print("done.")


if __name__ == "__main__":
    main()
