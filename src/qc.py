"""QC report: slice montage + basic stats -> PNG and self-contained HTML."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def volume_stats(data, mask=None):
    data = np.asarray(data, dtype=np.float32)
    vals = data if mask is None else data[np.asarray(mask) > 0]
    vals = vals[np.isfinite(vals)]
    mean = float(vals.mean())
    std = float(vals.std() + 1e-8)
    return {
        "shape": tuple(int(s) for s in data.shape),
        "mean": mean,
        "std": std,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "snr_estimate": float(mean / std),
    }


def save_montage(data, out_png, n_slices=9, cmap="gray"):
    """Save an axial montage PNG of evenly spaced slices."""
    data = np.asarray(data, dtype=np.float32)
    d = data.shape[2]
    idx = np.linspace(int(d * 0.15), int(d * 0.85), n_slices).astype(int)
    cols = 3
    rows = int(np.ceil(n_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    vmin, vmax = np.percentile(data, [1, 99])
    for ax, z in zip(axes, idx):
        ax.imshow(np.rot90(data[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def write_html_report(stats, montage_png, out_html, title="QC Report"):
    """Write a tiny self-contained HTML report (PNG referenced relatively)."""
    out_html = Path(out_html)
    rows = "\n".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in stats.items()
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:6px 12px}}
img{{max-width:100%}}</style></head>
<body><h1>{title}</h1>
<h2>Volume statistics</h2><table>{rows}</table>
<h2>Axial montage</h2><img src="{Path(montage_png).name}" alt="slice montage">
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html
