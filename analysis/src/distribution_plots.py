"""
Generate publication-quality distribution plots for reannotation label and bbox counts.
"""

from pathlib import Path
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['text.usetex'] = False

_LABEL_COLOR  = '#34A853'
_BBOX_COLOR   = '#3384FF'
_MEDIAN_COLOR = '#06d6a0'   # teal-green
_MEAN_COLOR   = '#e63946'   # crimson

_ALPHA = 0.75

_TICK_FS  = 18
_LABEL_FS = 20
_LEGEND_FS = 16


def _darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip('#')
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return '#{:02x}{:02x}{:02x}'.format(int(r * factor), int(g * factor), int(b * factor))


def _style_ax(ax) -> None:
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _value_counts(data: List[int]):
    arr = np.array(data)
    max_val = int(arr.max()) if len(arr) > 0 else 0
    values = np.arange(max_val + 1)
    counts = np.array([(arr == v).sum() for v in values])
    return values, counts, max_val


def _set_xticks(ax, max_val: int, step: int = 1) -> None:
    ax.set_xticks(range(0, max_val + 1, step))


def _plot_histogram(ax, data: List[int], color: str, xlabel: str, bare: bool = False,
                    xtick_step: int = 1) -> None:
    values, counts, max_val = _value_counts(data)
    bins = np.arange(-0.5, max_val + 1.5, 1.0)
    ax.hist(data, bins=bins, color=color, edgecolor='white', linewidth=0.6, alpha=_ALPHA)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    _set_xticks(ax, max_val, step=xtick_step)
    ax.tick_params(axis='both', labelsize=_TICK_FS, length=7, width=1.4)
    _style_ax(ax)

    mean_val   = float(np.mean(data))
    median_val = float(np.median(data))
    ax.axvline(mean_val,   color=_MEAN_COLOR,   linewidth=1.8, linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color=_MEDIAN_COLOR,  linewidth=1.8, linestyle=':',  label=f'Median: {median_val:.1f}')
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.9)

    if bare:
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        ax.set_xlabel(xlabel, fontsize=_LABEL_FS, fontfamily='serif')
        ax.set_ylabel("Number of images", fontsize=_LABEL_FS, fontfamily='serif')


def _plot_overlapping_histograms(ax, labels_data: List[int], bboxes_data: List[int]) -> None:
    bbox_color = _BBOX_COLOR
    arr_l = np.array(labels_data)
    arr_b = np.array(bboxes_data)
    max_val = max(int(arr_l.max()), int(arr_b.max()))
    values = np.arange(0, max_val + 1)
    counts_l = np.array([(arr_l == v).sum() for v in values])
    counts_b = np.array([(arr_b == v).sum() for v in values])

    w = 0.95
    ax.bar(values, counts_l, width=w, color=_LABEL_COLOR, edgecolor='white',
           linewidth=0.5, alpha=0.85, label='Labels')
    ax.bar(values, counts_b, width=w, color=bbox_color, edgecolor='white',
           linewidth=0.5, alpha=0.35, hatch='//', label='Bounding boxes')

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax.set_xticks(values[::2])
    ax.set_xlim(-0.6, max_val + 0.6)
    ax.tick_params(axis='both', labelsize=_TICK_FS, length=7, width=1.4)
    _style_ax(ax)

    for data, color, name in [
        (labels_data, _LABEL_COLOR, 'Labels'),
        (bboxes_data, bbox_color,   'BBoxes'),
    ]:
        mean_val   = float(np.mean(data))
        median_val = float(np.median(data))
        dark = _darken(color)
        ax.axvline(mean_val,   color=dark, linewidth=1.8, linestyle='--',
                   label=f'{name} mean: {mean_val:.2f}')
        ax.axvline(median_val, color=dark, linewidth=1.8, linestyle=':',
                   label=f'{name} median: {median_val:.1f}')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(fontsize=_LEGEND_FS - 1, framealpha=0.9)




def generate_distribution_plots(annotations, output_dir: Path) -> None:
    """
    Generate a single figure with two histogram subplots saved as distribution.png:
      - Left:  labels per image
      - Right: bounding boxes per image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_per_image: List[int] = [len(ann.reannotated_labels) for ann in annotations]
    bboxes_per_image: List[int] = [len(ann.bboxes) for ann in annotations]
    n = len(annotations)

    if n == 0:
        print("No annotations loaded — skipping distribution plots.")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _plot_overlapping_histograms(ax, labels_per_image, bboxes_per_image)
    plt.tight_layout()
    fig.savefig(output_dir / "distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    for stem, data, color, step in [
        ("distribution_labels", labels_per_image, _LABEL_COLOR, 1),
        ("distribution_bboxes", bboxes_per_image, _BBOX_COLOR,  2),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        _plot_histogram(ax, data, color, xlabel='', bare=True, xtick_step=step)
        plt.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\nDistribution plots saved to: {output_dir}")
    for name, data in [("Labels", labels_per_image), ("Bboxes", bboxes_per_image)]:
        arr = np.array(data)
        print(
            f"  {name:6s}/image — min: {arr.min()}, max: {arr.max()}, "
            f"mean: {arr.mean():.2f}, median: {np.median(arr):.1f}, "
            f"std: {arr.std():.2f}"
        )
