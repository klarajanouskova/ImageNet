#!/usr/bin/env python3
"""
Experiments
-----------
E1 — Overall model accuracy on ReImageNet annotations (ImGT / ReGT / Im∩Re + S/M/N breakdown):
    python run_analysis.py -p <predictions.csv|predictions.json>
    python run_analysis.py -p <knn.csv> --knn-k 11

E2 — Histogram of bbox count and label count per image (saved to data/plots/):
    python run_analysis.py --experiment dist

E3 — S-category accuracy overall + per attribute (crowd, ocr_needed, rendition, reflected):
    python run_analysis.py -p <predictions.csv|predictions.json> --experiment s-attr

E4 — Crop accuracy (ReGT only): overall / largest-bbox / any-crop / all-crops / excrops:
    python run_analysis.py -p <base_preds> --crop <crop_preds> --experiment crop
    Excrop file is auto-derived by replacing '-crop' with '-max-crop' in the crop filename.
    Alternatively, pass the max-crop file directly; the ordinary crop file is then derived by stripping 'max-'.

E5 — Derivative benchmark error estimation (GPT-5.4 ∩ SigLIP2-g agreement → ReGT vs benchmark GT):
    python run_analysis.py --experiment deriv-error
    Uses fixed files under data/benchmarks/siglip2/ and data/benchmarks/cgpt/.

E6 — Biggest-bbox accuracy (dominant-flag acc + single + cumulative modes, multilabel images):
    python run_analysis.py -p <predictions.csv|predictions.json> --experiment bbox-stats
    python run_analysis.py -p <knn.csv> --knn-k 11 --experiment bbox-stats
    Requires ANNOTATIONS_ROOT_FOLDER set in config/settings.py.

E7 — Accuracy by unique label count (single-label overall + M breakdown by 2, 3, 4, 5, ≥6):
    python run_analysis.py -p <predictions.csv|predictions.json> --experiment multilabel-stats
    python run_analysis.py -p <knn.csv> --knn-k 11 --experiment multilabel-stats

E8 — Unambiguous-label accuracy: single-label images, dominant-M images (unique dominant label), ExCrops:
    python run_analysis.py -p <predictions.csv|predictions.json> --experiment unambiguous-stats
    python run_analysis.py -p <knn.csv> --knn-k 11 --crop <crop_preds> --experiment unambiguous-stats
    GPT/Qwen models: omit --crop to get '-' for ExCrops.

E9 — Accuracy of model predictions vs prior annotation datasets (context_progress, real, multilabel, label_errors):
    python run_analysis.py --experiment prior-eval -p <predictions.csv|predictions.json>
    Reports acc vs original GT, proposed labels, and ReGT per split (S/S+/S-/M/M+/M-/N).
    Omit -p to see only group counts (no accuracy stats, no plots).

E10 — Alluvial diagrams showing group flow from prior annotation datasets to ReGT (saved to data/corrections/):
    python run_analysis.py --experiment prior-plots
    Reads correction CSVs from data/corrections/ (context_progress, real, multilabel, label_errors).
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.accuracy_calculator import AccuracyCalculator
from src.class_lookup_builder import ClassLookupBuilder
from src.distribution_plots import generate_distribution_plots
from src.utils import setup_logging
from config.settings import PROCESSED_DATA_DIR, PREDICTIONS_DIR, CSV_PREDICTIONS_DIR, BENCHMARKS_DIR, PLOTS_DIR


def _load_class_lookup():
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(
        repo_id="vrg-prague/ReImageNet",
        filename="class_update_config.json",
        repo_type="dataset",
    )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cl = ClassLookupBuilder()
    cl.build_lookup_from_json(cfg)
    return cl


def _print_results(name, acc_imgt, n_imgt, total_imgt,
                   acc_regt, n_regt, total_regt,
                   acc_imre, n_imre, total_imre,
                   group_accs):
    w = 72
    print("\n" + "=" * w)
    print(f"  {name}")
    print("=" * w)
    print(f"  ImGT  (vs original ImageNet labels):  {acc_imgt:6.2f}%  ({n_imgt}/{total_imgt})")
    print(f"  ReGT  (vs reannotated labels):         {acc_regt:6.2f}%  ({n_regt}/{total_regt})")
    print(f"  Im∩Re (correct under both):            {acc_imre:6.2f}%  ({n_imre}/{total_imre})")
    print("-" * w)
    print("  Image category breakdown (ReGT):")
    for gname, gacc, gcorrect, gtotal in group_accs:
        if gcorrect is not None:
            print(f"    {gname:5s}  {gacc:6.2f}%  ({gcorrect}/{gtotal})")
        else:
            print(f"    {gname:5s}  N/A  (0 images)")
    print("=" * w)


def _detect_pred_column(pred_path: Path, knn_k: int) -> str:
    """Return the prediction column name, auto-detecting KNN files."""
    import pandas as pd
    cols = pd.read_csv(pred_path, nrows=0).columns.tolist()
    if "top_1_pred" in cols:
        return "top_1_pred"
    knn_col = f"k_{knn_k}_pred"
    if knn_col in cols:
        print(f"KNN predictions detected — using k={knn_k} (column: '{knn_col}')")
        return knn_col
    raise ValueError(
        f"Neither 'top_1_pred' nor '{knn_col}' found in {pred_path.name}. "
        f"Available columns: {cols}"
    )


def _eval_csv(calculator, annotations, groups, pred_path, knn_k):
    pred_col = _detect_pred_column(pred_path, knn_k)
    csv_preds = calculator.load_csv_predictions(pred_path, pred_column=pred_col)

    acc_imgt, n_imgt, total_imgt = calculator.calculate_top1_csv_accuracy_vs_gt(
        csv_preds, annotations=annotations
    )
    acc_regt, n_regt, total_regt = calculator.calculate_top1_accuracy_vs_csv(
        annotations, csv_preds, return_counts=True
    )
    list_preds = {k: [v] for k, v in csv_preds.items()}
    acc_imre, n_imre, total_imre = calculator.calculate_im_intersect_re(annotations, list_preds)

    group_accs = []
    for gname in ["S", "S+", "S-", "M", "M+", "M-", "N"]:
        anns = groups.get(gname, [])
        if anns:
            gacc, gc, gt = calculator.calculate_top1_accuracy_vs_csv(anns, csv_preds, return_counts=True)
            group_accs.append((gname, gacc, gc, gt))
        else:
            group_accs.append((gname, 0.0, None, 0))

    return acc_imgt, n_imgt, total_imgt, acc_regt, n_regt, total_regt, acc_imre, n_imre, total_imre, group_accs


def _eval_json(calculator, annotations, groups, pred_path):
    with open(pred_path, encoding="utf-8") as f:
        preds_json = json.load(f)

    calculator.llm_image_subset = set(preds_json["predictions"].keys())
    list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)

    acc_imgt, n_imgt, total_imgt = calculator.calculate_top1_llm_accuracy_vs_gt(
        list_preds, annotations, evaluate_intersection_only=True
    )
    acc_regt, n_regt, total_regt = calculator.calculate_top1_accuracy_vs_csv(
        annotations, list_preds, evaluate_intersection_only=True, return_counts=True
    )
    acc_imre, n_imre, total_imre = calculator.calculate_im_intersect_re(annotations, list_preds)

    group_accs = []
    for gname in ["S", "S+", "S-", "M", "M+", "M-", "N"]:
        anns = groups.get(gname, [])
        if anns:
            gacc, gc, gt = calculator.calculate_top1_accuracy_vs_csv(
                anns, list_preds, evaluate_intersection_only=True, return_counts=True
            )
            group_accs.append((gname, gacc, gc, gt))
        else:
            group_accs.append((gname, 0.0, None, 0))

    return acc_imgt, n_imgt, total_imgt, acc_regt, n_regt, total_regt, acc_imre, n_imre, total_imre, group_accs


_ATTRIBUTE_ORDER = ["crowd", "reflected", "rendition", "ocr_needed", "mixed", "no_attributes"]
_ATTRIBUTE_LABELS = {
    "crowd":         "Crowd",
    "reflected":     "Reflection",
    "rendition":     "Rendition",
    "ocr_needed":    "Text",
    "mixed":         "Mix",
    "no_attributes": "No Attr",
}


def _eval_s_attributes(calculator, annotations, groups, pred_path, knn_k):
    suffix = pred_path.suffix.lower()
    if suffix == ".csv":
        pred_col = _detect_pred_column(pred_path, knn_k)
        raw_preds = calculator.load_csv_predictions(pred_path, pred_column=pred_col)
        list_preds = {k: [v] for k, v in raw_preds.items()}
    else:
        with open(pred_path, encoding="utf-8") as f:
            preds_json = json.load(f)
        calculator.llm_image_subset = set(preds_json["predictions"].keys())
        list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)

    s_anns = groups.get("S", [])
    s_acc, s_correct, s_total = calculator.calculate_top1_accuracy_vs_csv(
        s_anns, list_preds, evaluate_intersection_only=True, return_counts=True
    )

    # Group S images by bbox attribute
    attr_groups = {k: [] for k in _ATTRIBUTE_ORDER}
    ann_by_name = {a.image_name: a for a in s_anns}
    common_ids = set(ann_by_name.keys()) & set(list_preds.keys())

    for img_id in common_ids:
        ann = ann_by_name[img_id]
        image_attrs = set()
        has_mixed_bbox = False
        for bbox in ann.bboxes:
            bbox_attrs = [
                name for name, flag in [
                    ("crowd",      bbox.crowd_flag),
                    ("ocr_needed", bbox.ocr_needed_flag),
                    ("rendition",  bbox.rendition_flag),
                    ("reflected",  bbox.reflected_flag),
                ] if flag
            ]
            if len(bbox_attrs) > 1:
                has_mixed_bbox = True
            image_attrs.update(bbox_attrs)

        if has_mixed_bbox or len(image_attrs) > 1:
            attr_groups["mixed"].append(ann)
        elif len(image_attrs) == 0:
            attr_groups["no_attributes"].append(ann)
        else:
            attr_groups[list(image_attrs)[0]].append(ann)

    w = 72
    print("\n" + "=" * w)
    print(f"  {pred_path.name}  —  S-category attribute breakdown")
    print("=" * w)
    print(f"  S overall  {s_acc:6.2f}%  ({s_correct}/{s_total})")
    print("-" * w)
    for attr in _ATTRIBUTE_ORDER:
        anns = attr_groups[attr]
        if anns:
            acc, correct, total = calculator._calculate_accuracy_for_subset(anns, list_preds)
            print(f"  {_ATTRIBUTE_LABELS[attr]:20s}  {acc:6.2f}%  ({correct}/{total})")
        else:
            print(f"  {_ATTRIBUTE_LABELS[attr]:20s}  N/A  (0 images)")
    print("=" * w)


def _derive_excrop_path(crop_path):
    """Return the excrop path for a given crop path.

    If the crop path already contains '-max-crop' it is returned as-is;
    otherwise '-crop' is replaced with '-max-crop' in the stem.
    """
    if "-max-crop" in crop_path.stem:
        return crop_path
    excrop_stem = crop_path.stem.replace("-crop", "-max-crop", 1)
    return crop_path.parent / (excrop_stem + crop_path.suffix)


def _eval_crops(calculator, annotations, base_pred_path, crop_path, knn_k, class_lookup):
    if "-max-crop" in crop_path.stem:
        excrop_path = crop_path
        plain_stem = crop_path.stem.replace("-max-crop", "-crop", 1)
        crop_path = crop_path.parent / (plain_stem + crop_path.suffix)
    else:
        excrop_path = _derive_excrop_path(crop_path)

    # Value 1: overall ReGT on annotated images from base predictions
    suffix = base_pred_path.suffix.lower()
    if suffix == ".csv":
        pred_col = _detect_pred_column(base_pred_path, knn_k)
        base_preds = calculator.load_csv_predictions(base_pred_path, pred_column=pred_col)
        acc1, n1, total1 = calculator.calculate_top1_accuracy_vs_csv(
            annotations, base_preds, return_counts=True
        )
    else:
        with open(base_pred_path, encoding="utf-8") as f:
            preds_json = json.load(f)
        calculator.llm_image_subset = set(preds_json["predictions"].keys())
        list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)
        acc1, n1, total1 = calculator.calculate_top1_accuracy_vs_csv(
            annotations, list_preds, evaluate_intersection_only=True, return_counts=True
        )

    # Values 2-5: crop-based (class_lookup already set on calculator)
    crop_suffix = crop_path.suffix.lower()
    if crop_suffix == ".csv":
        regt2, n2, total2 = calculator.compute_largest_crop_accuracy(crop_path)
        regt3, n3, total3 = calculator.compute_crop_oracle_from_csv(crop_path, class_lookup=class_lookup)
        regt4, n4, total4 = calculator.compute_all_crops_accuracy(crop_path, class_lookup=class_lookup, no_fill=True)
        regt5, n5, total5 = calculator.compute_all_crops_accuracy(excrop_path, class_lookup=class_lookup, no_fill=True) if excrop_path.exists() else (None, None, None)
    else:
        regt2, n2, total2 = calculator.compute_largest_crop_accuracy_from_json(crop_path)
        regt3, n3, total3 = calculator.compute_crop_oracle_from_json(crop_path, class_lookup=class_lookup)
        regt4, n4, total4 = calculator.compute_all_crops_accuracy_from_json(crop_path, class_lookup=class_lookup, no_fill=True)
        regt5, n5, total5 = calculator.compute_all_crops_accuracy_from_json(excrop_path, class_lookup=class_lookup, no_fill=True) if excrop_path.exists() else (None, None, None)

    w = 72
    print("\n" + "=" * w)
    print(f"  {crop_path.name}  —  crop accuracy (ReGT)")
    print("=" * w)
    print(f"  Overall (annotated)    {acc1:6.2f}%  ({n1}/{total1})")
    print(f"  Largest bbox (50k)     {regt2:6.2f}%  ({n2}/{total2})" if regt2 is not None else "  Largest bbox (50k)    N/A  (mapping file missing)")
    print(f"  Any crop (50k)         {regt3:6.2f}%  ({n3}/{total3})")
    print(f"  All crops              {regt4:6.2f}%  ({n4}/{total4})")
    if regt5 is not None:
        print(f"  ExCrops                {regt5:6.2f}%  ({n5}/{total5})")
    else:
        print(f"  ExCrops                N/A  ({excrop_path.name} not found)")
    print("=" * w)


_BENCHMARKS = [
    ("v2",             "ImageNet-V2"),
    ("a",              "ImageNet-A"),
    ("r",              "ImageNet-R"),
    ("sketch",         "ImageNet-Sketch"),
    ("object_net",     "ObjectNet"),
    ("counter_animal", "CounterAnimal"),
]


def _eval_deriv_error(calculator, annotations, class_lookup):
    import ast
    import math
    import pandas as pd

    calculator.class_lookup = class_lookup
    ann_by_name = {a.image_name: a for a in annotations}
    deriv_root  = BENCHMARKS_DIR
    siglip2_dir = deriv_root / "siglip2"
    cgpt_dir    = deriv_root / "cgpt"

    # ── helpers ──────────────────────────────────────────────────────────────
    def _key_fn(ids_a, ids_b):
        """Pick the normalization function (full path vs basename) that maximizes ID overlap."""
        full_overlap = len(set(ids_a) & set(ids_b))
        base_a = {Path(x).name for x in ids_a}
        base_b = {Path(x).name for x in ids_b}
        base_overlap = len(base_a & base_b)
        return (lambda x: x) if full_overlap >= base_overlap else (lambda x: Path(x).name)

    def _norm(df, key):
        return {key(r["img_id"]): int(r["top_1_pred"]) for _, r in df.iterrows()}

    def _gt_map(df, key):
        return {key(r["img_id"]): r["original_label"] for _, r in df.iterrows()}

    def _parse_label(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return []
        try:
            return [int(val)]
        except (ValueError, TypeError):
            pass
        s = str(val).strip()
        if s.startswith("["):
            return [int(x) for x in ast.literal_eval(s)]
        if "," in s:
            return [int(x.strip()) for x in s.split(",")]
        return []

    def _agreed(pa, pb):
        common = set(pa) & set(pb)
        return {img_id: pa[img_id] for img_id in common if pa[img_id] == pb[img_id]}

    def _regt_error(agreed_preds):
        """Error rate vs ReImageNet reannotated labels."""
        correct = total = 0
        for img_id, pred in agreed_preds.items():
            ann = ann_by_name.get(img_id)
            if ann is None:
                continue
            total += 1
            reannotated = ann.reannotated_labels
            if not reannotated or (len(reannotated) == 1 and -1 in reannotated):
                correct += 1  # no valid label → cannot be wrong
            elif (class_lookup.is_in_equal_group_with_any(list(reannotated), pred)
                    if class_lookup else pred in reannotated):
                correct += 1
        acc = correct / total * 100 if total > 0 else 0.0
        return 100 - acc, total

    def _label_error(agreed_preds, gt_map):
        """Error rate vs benchmark's own original_label."""
        correct = total = 0
        for img_id, pred in agreed_preds.items():
            gts = _parse_label(gt_map.get(img_id))
            if not gts:
                continue
            total += 1
            if (class_lookup.is_in_equal_group_with_any(gts, pred)
                    if class_lookup else pred in gts):
                correct += 1
        acc = correct / total * 100 if total > 0 else 0.0
        return 100 - acc, total

    # ── load base IN1k predictions ───────────────────────────────────────────
    siglip2_path = CSV_PREDICTIONS_DIR / "siglip2-g.csv"
    gpt54_path   = PREDICTIONS_DIR / "gpt" / "gpt54-ow-siglip2_templated-cls_templated.json"

    df_s = pd.read_csv(siglip2_path)
    siglip2_in1k = dict(zip(df_s["img_id"], df_s["top_1_pred"].astype(int)))
    siglip2_cls  = dict(zip(df_s["img_id"], df_s["original_label"].astype(int)))

    with open(gpt54_path, encoding="utf-8") as f:
        gpt54_raw = json.load(f)["predictions"]
    gpt54_in1k = {img_id: int(v["label"]) for img_id, v in gpt54_raw.items()
                  if int(v["label"]) >= 0}

    # ── global IIN1k → εˆIN1k via ReGT ──────────────────────────────────────
    iin1k = _agreed(siglip2_in1k, gpt54_in1k)
    eps_in1k_global, _ = _regt_error(iin1k)

    # ── per benchmark ────────────────────────────────────────────────────────
    rows = []
    for key, bench_label in _BENCHMARKS:
        sp = siglip2_dir / f"siglip2-g_test_{key}_in1k.csv"
        cp = cgpt_dir    / f"cgpt-g_test_{key}_in1k.csv"
        if not (sp.exists() and cp.exists()):
            rows.append((bench_label, None))
            continue

        # class set for this benchmark (from derivative CSV labels)
        df_sp = pd.read_csv(sp)
        df_cp = pd.read_csv(cp)
        bench_classes = {cls for val in df_sp["original_label"] for cls in _parse_label(val)}

        # pick key function that maximizes ID overlap between the two model files
        kfn = _key_fn(df_sp["img_id"].tolist(), df_cp["img_id"].tolist())
        gt_map_bench = _gt_map(df_sp, kfn)

        # IIN1k restricted to benchmark classes → evaluate against ReGT
        iin1k_bench = {img_id: pred for img_id, pred in iin1k.items()
                       if siglip2_cls.get(img_id) in bench_classes}
        eps_in1k_bench, n_in1k_bench = _regt_error(iin1k_bench)

        # Ibench: agreement on derivative images (full IN1k head) → evaluate against benchmark GT
        ibench = _agreed(_norm(df_sp, kfn), _norm(df_cp, kfn))
        eps_bench, n_bench = _label_error(ibench, gt_map_bench)

        amp   = eps_bench / eps_in1k_bench if eps_in1k_bench > 0 else float("inf")
        n_err = (eps_bench - eps_in1k_bench) / 100 * n_bench

        rows.append((bench_label, {
            "n_imgs":       len(df_sp),
            "n_in1k_bench": n_in1k_bench,
            "eps_in1k":     eps_in1k_bench,
            "n_bench":      n_bench,
            "eps_bench":    eps_bench,
            "amp":          amp,
            "n_err":        n_err,
        }))

    # ── print ─────────────────────────────────────────────────────────────────
    w = 100
    print("\n" + "=" * w)
    print("  DERIVATIVE BENCHMARK ERROR ESTIMATION  (GPT-5.4 ∩ SigLIP2-g agreement)")
    print(f"  Global IIN1k: {len(iin1k):,} agreed  |  εˆIN1k = {eps_in1k_global:.2f}%")
    print()
    print("  Notation:")
    print("    # Imgs.  — total images in the derivative benchmark")
    print("    |IIN1k|  — IN1k val images where both models agree, restricted to benchmark classes")
    print("    εˆIN1k   — error rate of IIN1k vs ReGT  (N-category treated as correct)")
    print("    |Ibench| — derivative images where both models agree  (full IN1k head)")
    print("    εˆbench  — error rate of Ibench vs benchmark GT")
    print("    Amp      — εˆbench / εˆIN1k")
    print("    N̂_err   — (εˆbench − εˆIN1k) × |Ibench|  (lower bound on mislabeled images)")
    print("=" * w)
    print(f"  {'Benchmark':<20} {'# Imgs.':>8} {'|IIN1k|':>8} {'εˆIN1k':>8} {'|Ibench|':>9} {'εˆbench':>8} {'Amp':>5} {'N̂_err':>8}")
    print("-" * w)
    for bench_label, r in rows:
        if r is None:
            print(f"  {bench_label:<20}  (files missing)")
            continue
        amp_str = f"{r['amp']:.1f}×" if r["amp"] != float("inf") else "  inf"
        print(
            f"  {bench_label:<20} {r['n_imgs']:>8,} {r['n_in1k_bench']:>8,} {r['eps_in1k']:>7.2f}% "
            f"{r['n_bench']:>9,} {r['eps_bench']:>7.2f}% {amp_str:>5} {r['n_err']:>8.0f}"
        )
    print("=" * w)


def _eval_bbox_stats(calculator, annotations, pred_path, knn_k, image_base_path=None):
    import pandas as pd

    if pred_path.suffix.lower() == ".json":
        with open(pred_path, encoding="utf-8") as f:
            preds_json = json.load(f)
        list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)
    else:
        col = _detect_pred_column(pred_path, knn_k)
        df = pd.read_csv(pred_path)
        csv_preds = dict(zip(df["img_id"], df[col].astype(int)))
        list_preds = {img_id: [v] for img_id, v in csv_preds.items()}

    multilabel_names = {
        ann.image_name
        for ann in annotations
        if len({l for l in ann.reannotated_labels if l != -1}) > 1
    }

    single_ret = calculator.calculate_biggest_bbox_stats(
        annotations, list_preds, image_base_path, mode="single", max_k=5
    )
    cumul_ret = calculator.calculate_biggest_bbox_stats(
        annotations, list_preds, image_base_path, mode="cumulative", max_k=5
    )

    def _acc(ret, k):
        r = ret["k"][k]
        return f"{r['correct'] / r['total'] * 100:.2f}" if r["total"] > 0 else "n/a"

    # M = standard ReGT accuracy on all multilabel images (all reannotated labels as GT)
    m_correct = m_total = 0
    for ann in annotations:
        if ann.image_name not in multilabel_names:
            continue
        preds = list_preds.get(ann.image_name)
        if not preds or preds[0] == -1:
            continue
        reannotated = {l for l in ann.reannotated_labels if l != -1}
        if not reannotated:
            continue
        m_total += 1
        if calculator.is_correct(list(reannotated), preds[0]):
            m_correct += 1
    m_acc = f"{m_correct / m_total * 100:.2f}" if m_total > 0 else "n/a"

    dom = f"{single_ret['dom_acc']:.2f}"
    s   = [_acc(single_ret, k) for k in range(1, 6)]
    c   = [_acc(cumul_ret,  k) for k in range(2, 6)]

    col = 7
    sep = "  "
    gap = "    "   # extra gap between single and cumul groups

    def _row(cells, between=None):
        parts = [f"{v:<{col}}" for v in cells]
        if between is not None:
            parts.insert(between, gap)
        return "  " + sep.join(parts)

    headers = ["Dom", "k=1", "k=2", "k=3", "k=4", "k=5", "≤2", "≤3", "≤4", "≤5", "M"]
    data    = [dom] + s + c + [m_acc]

    w = len(_row(headers, between=6)) + 2
    print("\n" + "=" * w)
    print("  BIGGEST BBOX ACCURACY  (multilabel images only)")
    print(_row(headers, between=6))
    print(_row(data,    between=6))
    print("=" * w)


def _eval_unambiguous_stats(calculator, annotations, pred_path, knn_k, crop_path=None):
    """E8 — accuracy on images with an unambiguous label: single-label, dominant-M, excrop."""
    suffix = pred_path.suffix.lower()
    if suffix == ".csv":
        pred_col = _detect_pred_column(pred_path, knn_k)
        raw_preds = calculator.load_csv_predictions(pred_path, pred_column=pred_col)
        list_preds = {k: [v] for k, v in raw_preds.items()}
    else:
        with open(pred_path, encoding="utf-8") as f:
            preds_json = json.load(f)
        calculator.llm_image_subset = set(preds_json["predictions"].keys())
        list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)

    ann_by_name = {a.image_name: a for a in annotations}
    common_ids = set(ann_by_name.keys()) & set(list_preds.keys())

    single_anns = []
    dominant_m_anns = []

    for img_id in common_ids:
        ann = ann_by_name[img_id]
        unique_labels = {lbl for lbl in ann.reannotated_labels if lbl != -1}
        if calculator.class_lookup is not None:
            seen: set = set()
            num_unique = 0
            for lbl in unique_labels:
                if lbl not in seen:
                    num_unique += 1
                    seen.add(lbl)
                    seen.update(calculator.class_lookup.get_neighbors(lbl))
        else:
            num_unique = len(unique_labels)

        if num_unique == 1:
            single_anns.append(ann)
        elif num_unique >= 2:
            dominant_labels = {
                lbl
                for bbox in ann.bboxes
                if bbox.dominant_object
                for lbl in bbox.labels
                if lbl != -1
            }
            if len(dominant_labels) == 1:
                dominant_m_anns.append(ann)

    def _row(label, anns):
        if anns:
            acc, correct, total = calculator._calculate_accuracy_for_subset(anns, list_preds)
            return f"  {label:<20}  {acc:6.2f}%  ({correct}/{total})"
        return f"  {label:<20}  N/A  (0 images)"

    if crop_path is not None:
        excrop_path = _derive_excrop_path(crop_path)
        if excrop_path.exists():
            regt_excrop, n_correct, n_total = calculator.compute_all_crops_accuracy(
                excrop_path, class_lookup=calculator.class_lookup, no_fill=True
            )
            excrop_str = f"  {'ExCrops':<20}  {regt_excrop:6.2f}%  ({n_correct}/{n_total})"
        else:
            excrop_str = f"  {'ExCrops':<20}  N/A  ({excrop_path.name} not found)"
    else:
        excrop_str = f"  {'ExCrops':<20}  -"

    w = 72
    print("\n" + "=" * w)
    print(f"  {pred_path.name}  —  unambiguous-label accuracy")
    print("=" * w)
    print(_row("Single-label", single_anns))
    print(_row("Dominant-M", dominant_m_anns))
    print(excrop_str)
    print("=" * w)


def _eval_multilabel_stats(calculator, annotations, pred_path, knn_k):
    from collections import defaultdict

    suffix = pred_path.suffix.lower()
    if suffix == ".csv":
        pred_col = _detect_pred_column(pred_path, knn_k)
        raw_preds = calculator.load_csv_predictions(pred_path, pred_column=pred_col)
        list_preds = {k: [v] for k, v in raw_preds.items()}
    else:
        with open(pred_path, encoding="utf-8") as f:
            preds_json = json.load(f)
        calculator.llm_image_subset = set(preds_json["predictions"].keys())
        list_preds, _, _ = calculator._convert_llm_json_to_predictions(preds_json)

    ann_by_name = {a.image_name: a for a in annotations}
    common_ids = set(ann_by_name.keys()) & set(list_preds.keys())

    label_count_groups: dict = defaultdict(list)
    n_count = 0

    for img_id in common_ids:
        ann = ann_by_name[img_id]
        unique_labels = {lbl for lbl in ann.reannotated_labels if lbl != -1}
        if calculator.class_lookup is not None:
            seen: set = set()
            num_unique = 0
            for lbl in unique_labels:
                if lbl not in seen:
                    num_unique += 1
                    seen.add(lbl)
                    seen.update(calculator.class_lookup.get_neighbors(lbl))
        else:
            num_unique = len(unique_labels)

        if num_unique > 0:
            label_count_groups[num_unique].append(ann)
        else:
            n_count += 1

    def _row(label, anns):
        if anns:
            acc, correct, total = calculator._calculate_accuracy_for_subset(anns, list_preds)
            return f"  {label:<20}  {acc:6.2f}%  ({correct}/{total})"
        return f"  {label:<20}  N/A  (0 images)"

    w = 72
    print("\n" + "=" * w)
    print(f"  {pred_path.name}  —  accuracy by unique label count")
    print("=" * w)
    print(_row("Single-label (1)", label_count_groups.get(1, [])))
    print("-" * w)
    print("  M images by label count:")
    for n in range(2, 6):
        print(_row(f"  {n} labels", label_count_groups.get(n, [])))
    anns_6plus = [ann for k, anns in label_count_groups.items() if k >= 6 for ann in anns]
    print(_row("  ≥6 labels", anns_6plus))
    print(f"\n  N images (no valid labels): {n_count}")
    print("=" * w)


def _resolve_path(p: Path, base_dir: Path, crop_subdir: str = "crop") -> Path:
    """Resolve a user path against base_dir when given as a bare filename.

    Search order:
      1. Absolute path → use as-is.
      2. Relative path that exists from cwd → use as-is.
      3. base_dir / p
      4. base_dir / csv / p  (CSV predictions subfolder)
      5. base_dir / crop_subdir / p  (only when crop_subdir is given)
    Returns a resolved absolute Path (may not exist; caller checks).
    """
    candidate = p.expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    direct = base_dir / p
    if direct.exists():
        return direct.resolve()
    via_csv = base_dir / "csv" / p
    if via_csv.exists():
        return via_csv.resolve()
    if crop_subdir:
        via_crop = base_dir / crop_subdir / p
        if via_crop.exists():
            return via_crop.resolve()
    return direct.resolve()


_EXPERIMENT_ALIASES = {
    "e1": "acc", "e2": "dist", "e3": "s-attr", "e4": "crop",
    "e5": "deriv-error", "e6": "bbox-stats", "e7": "multilabel-stats",
    "e8": "unambiguous-stats", "e9": "prior-eval", "e10": "prior-plots",
}


def _resolve_experiment(value: str) -> str:
    return _EXPERIMENT_ALIASES.get(value.lower(), value)


def main():
    parser = argparse.ArgumentParser(
        description="Paper supplementary experiments"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=_resolve_experiment,
        choices=["acc", "dist", "s-attr", "crop", "deriv-error", "bbox-stats", "multilabel-stats", "unambiguous-stats", "prior-eval", "prior-plots"],
        default="acc",
        help="Experiment to run: 'acc'/E1, 'dist'/E2, 's-attr'/E3, 'crop'/E4, 'deriv-error'/E5, 'bbox-stats'/E6, 'multilabel-stats'/E7, 'unambiguous-stats'/E8, 'prior-eval'/E9, 'prior-plots'/E10",
    )
    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        help=(
            "[E1, E3, E4, E6, E7, E8, E9] Predictions file (.csv with img_id/top_1_pred, "
            "or LLM .json with predictions dict). "
            "Bare filenames are resolved against data/predictions/ (or --data-dir)."
        ),
    )
    parser.add_argument(
        "--crop",
        type=Path,
        help=(
            "[E4, E8] Crop predictions file (CSV or JSON). "
            "Bare filenames are resolved against data/predictions/crop/ (or --data-dir/crop/). "
            "ExCrop file is auto-derived by replacing '-crop' with '-max-crop' in the filename. "
            "Alternatively, pass the max-crop file directly; the ordinary crop file is then derived by stripping 'max-' (E4 only)."
        ),
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=11,
        help="[E1, E4, E6] For KNN CSV files: which k to use (default: 11). Ignored for standard CSVs.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity (default: WARNING)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to search for prediction files when a bare filename is given "
            "(default: data/predictions relative to project root). "
            "Crop files are also searched under DIR/crop/."
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "[E6] Path to ImageNet validation images root (the directory containing "
            "per-class subfolders). Overrides the IMAGENET_VAL_PATH environment variable "
            "and the fallback in config/settings.py."
        ),
    )
    args = parser.parse_args()

    pred_base = args.data_dir.expanduser().resolve() if args.data_dir else PREDICTIONS_DIR
    if args.predictions:
        args.predictions = _resolve_path(args.predictions, pred_base)
    if args.crop:
        args.crop = _resolve_path(args.crop, pred_base)

    setup_logging(args.log_level)

    print("Loading annotations from HuggingFace…")
    calculator = AccuracyCalculator(PROCESSED_DATA_DIR)
    annotations = calculator.load_annotations()

    if args.experiment == "dist":
        generate_distribution_plots(annotations, PLOTS_DIR)
        return

    if args.experiment == "prior-eval":
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        csv_path = None
        llm_json_path = None
        if args.predictions:
            pred_path = args.predictions
            if not pred_path.exists():
                print(f"Error: file not found: {pred_path}", file=sys.stderr)
                sys.exit(1)
            if pred_path.suffix.lower() == ".json":
                llm_json_path = pred_path
            else:
                csv_path = pred_path
        calculator.evaluate_corrections(
            csv_path=csv_path, class_lookup=class_lookup, llm_json_path=llm_json_path,
            generate_plots=False,
        )
        return

    if args.experiment == "prior-plots":
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        calculator.evaluate_corrections(
            csv_path=None, class_lookup=class_lookup, llm_json_path=None,
            generate_plots=True,
        )
        return

    if args.experiment == "deriv-error":
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        _eval_deriv_error(calculator, annotations, class_lookup)
        return

    # E6 — bbox stats
    if args.experiment == "bbox-stats":
        if not args.predictions:
            parser.error("--predictions / -p is required for the 'bbox-stats' experiment")
        pred_path = args.predictions
        if not pred_path.exists():
            print(f"Error: file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        image_dir = str(args.image_dir.expanduser().resolve()) if args.image_dir else None
        _eval_bbox_stats(calculator, annotations, pred_path, args.knn_k, image_base_path=image_dir)
        return

    # E7 — multilabel stats
    if args.experiment == "multilabel-stats":
        if not args.predictions:
            parser.error("--predictions / -p is required for the 'multilabel-stats' experiment")
        pred_path = args.predictions
        if not pred_path.exists():
            print(f"Error: file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        _eval_multilabel_stats(calculator, annotations, pred_path, args.knn_k)
        return

    # E8 — unambiguous-label stats
    if args.experiment == "unambiguous-stats":
        if not args.predictions:
            parser.error("--predictions / -p is required for the 'unambiguous-stats' experiment")
        pred_path = args.predictions
        if not pred_path.exists():
            print(f"Error: file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        crop_path = None
        if args.crop:
            crop_path = args.crop
            if not crop_path.exists():
                print(f"Error: file not found: {crop_path}", file=sys.stderr)
                sys.exit(1)
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        _eval_unambiguous_stats(calculator, annotations, pred_path, args.knn_k, crop_path=crop_path)
        return

    # E4 — crop experiment
    if args.experiment == "crop":
        if not args.predictions or not args.crop:
            parser.error("--predictions / -p and --crop are both required for the 'crop' experiment")
        pred_path = args.predictions
        crop_path = args.crop
        for p in (pred_path, crop_path):
            if not p.exists():
                print(f"Error: file not found: {p}", file=sys.stderr)
                sys.exit(1)
        print("Loading class equivalence config from HuggingFace…")
        class_lookup = _load_class_lookup()
        calculator.class_lookup = class_lookup
        _eval_crops(calculator, annotations, pred_path, crop_path, args.knn_k, class_lookup)
        return

    # E1 / E3 — require predictions file
    if not args.predictions:
        parser.error(f"--predictions / -p is required for the '{args.experiment}' experiment")

    pred_path = args.predictions
    if not pred_path.exists():
        print(f"Error: file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading class equivalence config from HuggingFace…")
    class_lookup = _load_class_lookup()

    print("Categorizing images…")
    calculator.class_lookup = class_lookup
    groups = calculator.categorize_images(annotations)

    if args.experiment == "s-attr":
        _eval_s_attributes(calculator, annotations, groups, pred_path, args.knn_k)
        return

    # E1 — accuracy
    suffix = pred_path.suffix.lower()
    if suffix == ".csv":
        results = _eval_csv(calculator, annotations, groups, pred_path, args.knn_k)
    elif suffix == ".json":
        results = _eval_json(calculator, annotations, groups, pred_path)
    else:
        print(f"Error: unsupported extension '{suffix}', expected .csv or .json", file=sys.stderr)
        sys.exit(1)

    _print_results(pred_path.name, *results)


if __name__ == "__main__":
    main()
