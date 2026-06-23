from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import imagenet_classes
from src.data_models import BoundingBox, ImageAnnotation, AccuracyResults


class StatsAnalyzerMixin:
    """Mixin providing stats analysis and visualization for AccuracyCalculator."""

    def _get_annotation_group(self, ann: "ImageAnnotation") -> str:
        """Categorize a single annotation into N / S+ / S- / M+ / M- using the same logic as categorize_images."""
        labels = ann.reannotated_labels
        n = len(labels)
        if n == 0:
            return "N"
        if n == 1:
            if -1 in labels:
                return "N"
            in_rean = self._is_original_in_reannotated(ann.original_class, labels)
            return "S+" if in_rean else "S-"
        # n > 1
        if labels == {-1}:
            return "N"
        valid = labels - {-1}
        if len(valid) == 1:
            in_rean = self._is_original_in_reannotated(ann.original_class, valid)
            return "S+" if in_rean else "S-"
        in_rean = self._is_original_in_reannotated(ann.original_class, labels)
        return "M+" if in_rean else "M-"

    def evaluate_corrections(
        self,
        csv_path: Optional[Path],
        class_lookup,
        llm_json_path: Optional[Path] = None,
        generate_plots: bool = True,
    ) -> None:
        """Evaluate model predictions against the 4 correction CSV files in data/prior_corrections/.

        For each file (context_progress, real, multilabel, label_errors):
          - proposed_labels is treated as the reannotated GT for that image
          - Empty proposed_labels → N category (auto-correct)
          - Reports: group counts, acc vs original ImageNet GT, acc vs proposed labels,
            and per-split (N/S/S+/S-/M/M+/M-) accuracy vs proposed labels.
        """
        self.class_lookup = class_lookup

        # Load standard reannotated annotations once for per-file subset evaluation
        all_annotations = self.load_annotations()
        ann_by_id: Dict[str, "ImageAnnotation"] = {a.image_name: a for a in all_annotations}

        # Load model predictions
        csv_predictions: Optional[Dict[str, List[int]]] = None
        if llm_json_path is not None:
            with open(llm_json_path, 'r', encoding='utf-8') as f:
                preds_json = json.load(f)
            csv_predictions, _, _ = self._convert_llm_json_to_predictions(preds_json)
            print(f"Loaded LLM predictions: {len(csv_predictions)} images from {llm_json_path.name}")
        elif csv_path is not None:
            raw_preds = self.load_csv_predictions(csv_path)
            csv_predictions = {img_id: [pred] for img_id, pred in raw_preds.items()}
            print(f"Loaded CSV predictions: {len(csv_predictions)} images from {csv_path.name}")

        corrections_dir = Path(project_root) / "data" / "corrections"
        correction_files = [
            "real.csv",
            "multilabel.csv",
            "context_progress.csv",
            "label_errors.csv",
        ]

        for fname in correction_files:
            fpath = corrections_dir / fname
            if not fpath.exists():
                print(f"Warning: {fname} not found, skipping")
                continue

            df = pd.read_csv(fpath, dtype=str)

            # Build proposed_labels dict: img_id -> set[int] (empty = N / auto-correct)
            proposed_by_id: Dict[str, Set[int]] = {}
            for _, row in df.iterrows():
                img_id = str(row['id']).strip()
                raw = str(row.get('proposed_labels', '')).strip()
                if raw == '' or raw.lower() == 'nan':
                    proposed_by_id[img_id] = set()
                else:
                    labels: Set[int] = set()
                    for v in raw.split(','):
                        v = v.strip()
                        if v:
                            try:
                                labels.add(int(float(v)))
                            except ValueError:
                                pass
                    proposed_by_id[img_id] = labels

            # Categorize images by proposed_labels and original GT
            groups: Dict[str, List[str]] = {
                'N': [], 'S': [], 'S+': [], 'S-': [],
                'M': [], 'M+': [], 'M-': [],
            }
            gt_by_id: Dict[str, int] = {}
            for img_id, proposed in proposed_by_id.items():
                gt_class = imagenet_classes.val_image_to_1k_label(img_id)
                if not isinstance(gt_class, int) or gt_class not in range(1000):
                    continue
                gt_by_id[img_id] = gt_class

                n_labels = len(proposed)
                original_in_proposed = self._is_original_in_reannotated(gt_class, proposed)

                if n_labels == 0:
                    groups['N'].append(img_id)
                elif n_labels == 1:
                    groups['S'].append(img_id)
                    if original_in_proposed:
                        groups['S+'].append(img_id)
                    else:
                        groups['S-'].append(img_id)
                else:
                    groups['M'].append(img_id)
                    if original_in_proposed:
                        groups['M+'].append(img_id)
                    else:
                        groups['M-'].append(img_id)

            total = sum(len(v) for v in [groups['N'], groups['S'], groups['M']])

            print("\n" + "=" * 80)
            print(f"CORRECTIONS FILE: {fname}  ({total} images)")
            print("=" * 80)

            if csv_predictions is None:
                print("  (no model predictions loaded — skipping accuracy stats)")
            else:
                # Helper: accuracy of csv_predictions for a list of image ids, against proposed_labels
                def _acc_vs_proposed(img_ids: List[str]) -> Tuple[float, int, int]:
                    correct = 0
                    evaluated = 0
                    for img_id in img_ids:
                        preds = csv_predictions.get(img_id)
                        if preds is None:
                            continue
                        evaluated += 1
                        proposed = proposed_by_id.get(img_id, set())
                        if len(proposed) == 0:
                            correct += 1  # N = auto-correct
                            continue
                        for p in preds:
                            if self.is_correct(list(proposed), p):
                                correct += 1
                                break
                    acc = round(correct / evaluated * 100, 2) if evaluated > 0 else 0.0
                    return acc, correct, evaluated

                # Helper: accuracy against standard reannotated GT for a subset of image ids
                def _acc_vs_regt(img_ids: List[str]) -> Tuple[float, int, int]:
                    subset_anns = [ann_by_id[img_id] for img_id in img_ids if img_id in ann_by_id]
                    if not subset_anns:
                        return 0.0, 0, 0
                    prev_level = self.logger.level
                    self.logger.setLevel(logging.WARNING)
                    try:
                        acc, correct, total = self.calculate_top1_accuracy_vs_csv(
                            subset_anns, csv_predictions, evaluate_intersection_only=True, return_counts=True
                        )
                    finally:
                        self.logger.setLevel(prev_level)
                    return acc, correct, total

                # Accuracy vs original ImageNet GT for images in this file
                gt_correct = 0
                gt_evaluated = 0
                for img_id, gt_class in gt_by_id.items():
                    preds = csv_predictions.get(img_id)
                    if preds is None:
                        continue
                    gt_evaluated += 1
                    for p in preds:
                        if self.is_correct(gt_class, p):
                            gt_correct += 1
                            break
                acc_gt = round(gt_correct / gt_evaluated * 100, 2) if gt_evaluated > 0 else 0.0

                # Overall accuracy vs proposed labels (all images: N + S + M)
                all_ids = groups['N'] + groups['S'] + groups['M']
                acc_prop, prop_correct, prop_evaluated = _acc_vs_proposed(all_ids)

                # Per-split accuracy vs proposed labels and vs reannotated GT
                print(f"\n  Per-split accuracy vs proposed labels / reannotated GT:")
                split_order = ['S', 'S+', 'S-', 'M', 'M+', 'M-', 'N']
                split_accs_prop: Dict[str, float] = {}
                split_accs_regt: Dict[str, float] = {}
                for split in split_order:
                    ids = groups[split]
                    if not ids:
                        print(f"    {split:<4}: {'—':>7}  /  {'—':>7}  (0 images)")
                        split_accs_prop[split] = 0.0
                        split_accs_regt[split] = 0.0
                        continue
                    acc_s, c_s, e_s = _acc_vs_proposed(ids)
                    acc_r, c_r, e_r = _acc_vs_regt(ids)
                    split_accs_prop[split] = acc_s
                    split_accs_regt[split] = acc_r
                    regt_str = f"{acc_r:6.2f}% ({c_r}/{e_r})" if e_r > 0 else "      —"
                    print(f"    {split:<4}: {acc_s:6.2f}%  /  {regt_str}  ({c_s}/{e_s} evaluated, {len(ids)} total)")

                # Overall acc vs reannotated for the full subset
                subset_anns = [ann_by_id[img_id] for img_id in gt_by_id if img_id in ann_by_id]
                if subset_anns:
                    acc_reann, reann_correct, reann_total = self.calculate_top1_accuracy_vs_csv(
                        subset_anns, csv_predictions, evaluate_intersection_only=True, return_counts=True
                    )
                    latex_prop = [acc_gt, acc_prop] + [split_accs_prop[s] for s in ['S', 'S+', 'S-', 'M', 'M+', 'M-']]
                    latex_regt = [acc_gt, acc_reann] + [split_accs_regt[s] for s in ['S', 'S+', 'S-', 'M', 'M+', 'M-']]
                    print(f"\n  LaTeX proposed (GT & proposed & S & S+ & S- & M & M+ & M-):")
                    print("  " + " & ".join(f"{v:.2f}" for v in latex_prop))
                    print(f"\n  LaTeX reannotated GT (GT & reannotated & S & S+ & S- & M & M+ & M-):")
                    print("  " + " & ".join(f"{v:.2f}" for v in latex_regt))
                    print(f"\n  LaTeX summary (GT & annotations & ReGT):")
                    print(f"  {acc_gt:.2f} & {acc_prop:.2f} & {acc_reann:.2f} \\\\")
                else:
                    print(f"\n  (no reannotated annotations found for this subset)")
                    latex_vals = [acc_gt, acc_prop] + [split_accs_prop[s] for s in ['S', 'S+', 'S-', 'M', 'M+', 'M-']]
                    print(f"\n  LaTeX (GT & proposed & S & S+ & S- & M & M+ & M-):")
                    print("  " + " & ".join(f"{v:.2f}" for v in latex_vals))
                print(80 * "-")

            if generate_plots:
                # Alluvial diagram: correction groups (left) vs. ReGT annotation groups (right)
                flows: Dict[Tuple[str, str], int] = defaultdict(int)
                for img_id in proposed_by_id:
                    corr_group = next(
                        (g for g in ('N', 'S+', 'S-', 'M+', 'M-') if img_id in groups[g]),
                        None,
                    )
                    if corr_group is None:
                        continue
                    regt_group = self._get_annotation_group(ann_by_id[img_id]) if img_id in ann_by_id else "No ann."
                    flows[(corr_group, regt_group)] += 1

                plot_stem = Path(fname).stem
                _CORRECTION_DISPLAY_NAMES = {
                    "real":             "ImageNet ReaL",
                    "multilabel":       "Multi-label annotations",
                    "context_progress": "ImageNetMultiLabel",
                    "label_errors":     "Label Errors",
                }
                display_name = _CORRECTION_DISPLAY_NAMES.get(plot_stem, plot_stem)
                plot_path = corrections_dir / f"alluvial_{plot_stem}.png"
                _plot_alluvial_corrections(
                    flows=dict(flows),
                    left_title=display_name,
                    right_title="ReImageNet",
                    output_path=plot_path,
                    title=f"{display_name} vs. ReGT",
                )


def _plot_alluvial_corrections(
    flows: Dict[Tuple[str, str], int],
    left_title: str,
    right_title: str,
    output_path: Path,
    title: str = "Label group flow",
) -> None:
    """Draw an alluvial (Sankey-style) diagram comparing two group categorisations.

    Each image maps to a left group (correction-based) and a right group
    (ReGT annotation-based).  Stacked bars on each side are connected by filled
    Bezier ribbons whose width is proportional to the flow count.
    """
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    import matplotlib as mpl

    # Paper-appropriate serif font for all text in this figure
    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "text.usetex": False,
    }):
        # Google / Material Design palette
        GROUP_ORDER = ["N", "S+", "S-", "M+", "M-", "No ann."]
        GROUP_COLORS = {
            "N":       "#757575",   # Material Grey 600
            "S+":      "#34A853",   # Google Green
            "S-":      "#EA4335",   # Google Red
            "M+":      "#4285F4",   # Google Blue
            "M-":      "#9C27B0",   # Material Purple 600
            "No ann.": "#FF9800",   # Material Orange 600
        }
        ALPHA_RIBBON = 0.35
        BAR_WIDTH = 0.055
        LEFT_X, RIGHT_X = 0.20, 0.80
        GAP_FRACTION = 0.025

        # Collect groups present in the data
        left_groups  = [g for g in GROUP_ORDER if any(lg == g for lg, _ in flows)]
        right_groups = [g for g in GROUP_ORDER if any(rg == g for _, rg in flows)]

        left_totals  = {g: sum(v for (lg, rg), v in flows.items() if lg == g) for g in left_groups}
        right_totals = {g: sum(v for (lg, rg), v in flows.items() if rg == g) for g in right_groups}

        grand_total = sum(left_totals.values())
        if grand_total == 0:
            print("  (no flow data — skipping alluvial plot)")
            return

        gap = grand_total * GAP_FRACTION

        def _bar_positions(groups: List[str], totals: Dict[str, int]) -> Dict[str, float]:
            pos: Dict[str, float] = {}
            y = 0.0
            for g in groups:
                pos[g] = y
                y += totals[g] + gap
            return pos

        left_pos  = _bar_positions(left_groups,  left_totals)
        right_pos = _bar_positions(right_groups, right_totals)

        left_off:  Dict[str, float] = {g: 0.0 for g in left_groups}
        right_off: Dict[str, float] = {g: 0.0 for g in right_groups}

        # Sort to minimise ribbon crossings
        sorted_flows = sorted(
            flows.items(),
            key=lambda kv: (
                GROUP_ORDER.index(kv[0][0]) if kv[0][0] in GROUP_ORDER else 99,
                GROUP_ORDER.index(kv[0][1]) if kv[0][1] in GROUP_ORDER else 99,
            ),
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        # Draw ribbons
        for (lg, rg), count in sorted_flows:
            if count == 0 or lg not in left_pos or rg not in right_pos:
                continue

            yl_bot = left_pos[lg]  + left_off[lg];  yl_top = yl_bot + count;  left_off[lg]  += count
            yr_bot = right_pos[rg] + right_off[rg]; yr_top = yr_bot + count;  right_off[rg] += count

            x0 = LEFT_X  + BAR_WIDTH / 2
            x1 = RIGHT_X - BAR_WIDTH / 2
            cx = (x0 + x1) / 2

            verts = [
                [x0, yl_bot], [cx, yl_bot], [cx, yr_bot], [x1, yr_bot],
                [x1, yr_top], [cx, yr_top], [cx, yl_top], [x0, yl_top],
                [x0, yl_bot],
            ]
            codes = [
                MPath.MOVETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.LINETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.CLOSEPOLY,
            ]
            ax.add_patch(PathPatch(
                MPath(verts, codes),
                facecolor=GROUP_COLORS.get(lg, "#aaaaaa"),
                alpha=ALPHA_RIBBON,
                edgecolor="none",
                zorder=2,
            ))

        # Draw bars and group labels
        for g in left_groups:
            y0, h = left_pos[g], left_totals[g]
            ax.barh(y0 + h / 2, BAR_WIDTH, height=h, left=LEFT_X - BAR_WIDTH / 2,
                    color=GROUP_COLORS.get(g, "#aaaaaa"), zorder=5, align="center",
                    edgecolor="white", linewidth=0.4)
            ax.text(LEFT_X - BAR_WIDTH / 2 - 0.013, y0 + h / 2,
                    f"{g}  ({h:,})", ha="right", va="center", fontsize=28,
                    color="#222222")

        for g in right_groups:
            y0, h = right_pos[g], right_totals[g]
            ax.barh(y0 + h / 2, BAR_WIDTH, height=h, left=RIGHT_X - BAR_WIDTH / 2,
                    color=GROUP_COLORS.get(g, "#aaaaaa"), zorder=5, align="center",
                    edgecolor="white", linewidth=0.4)
            ax.text(RIGHT_X + BAR_WIDTH / 2 + 0.013, y0 + h / 2,
                    f"({h:,})  {g}", ha="left", va="center", fontsize=28,
                    color="#222222")

        max_y = max(
            max(left_pos[g]  + left_totals[g]  for g in left_groups),
            max(right_pos[g] + right_totals[g] for g in right_groups),
        )

        ax.text(LEFT_X,  max_y + gap * 4.0, left_title,  ha="center", va="bottom",
                fontsize=34, fontweight="bold", color="#111111")
        ax.text(RIGHT_X, max_y + gap * 4.0, right_title, ha="center", va="bottom",
                fontsize=34, fontweight="bold", color="#111111")

        ax.set_xlim(0, 1)
        ax.set_ylim(-gap, max_y + gap * 4)
        ax.set_title("")
        ax.axis("off")

        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close()
    print(f"  Alluvial plot saved: {output_path}")


