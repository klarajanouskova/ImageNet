"""Top-1 accuracy with original ImageNet labels (GT) and ReImageNet labels (ReGT).

Prints one row per prediction file with the columns of the paper's main table:
GT on A\\N, then ReGT on A\\N, S, S+, S-, M, M+, M-. Images in N (no valid
ImageNet-1k class) are excluded.

Usage:
    python -m eval_reimagenet.evaluate --preds preds/eva02.csv preds/gpt54.json
    python -m eval_reimagenet.evaluate --preds preds/dinov3_knn.csv --pred-column k_11_pred
    python -m eval_reimagenet.evaluate --preds preds/*.csv --format latex
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from .annotations import DEFAULT_DATA_DIR, load_reimagenet
from .metrics import COMPOSITE, evaluate
from .predictions import load_predictions


def _fmt(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", required=True, type=Path, help="Prediction files (.csv or .json)")
    parser.add_argument("--names", nargs="+", help="Row names, one per prediction file (default: file stem)")
    parser.add_argument("--pred-column", default=None, help="CSV prediction column (default: top_1_pred)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="ReImageNet release directory")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate only images that have a prediction instead of failing on missing ones",
    )
    parser.add_argument("--format", choices=["text", "csv", "latex"], default="text")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.preds):
        parser.error("--names needs one name per --preds file")
    names = args.names or [p.stem for p in args.preds]

    annotations, eq, _ = load_reimagenet(args.data_dir)

    rows = []
    for name, path in zip(names, args.preds):
        predictions = load_predictions(path, column=args.pred_column)
        result = evaluate(annotations, predictions, eq, allow_missing=args.allow_missing)
        if result.num_missing:
            print(f"warning: {name}: {result.num_missing} images without a prediction were skipped", file=sys.stderr)
        rows.append((name, result))

    columns = ["GT A\\N"] + [f"ReGT {s}" for s in COMPOSITE] + ["ReGT-GT A\\N"]
    sizes = rows[0][1].total
    size_line = ", ".join(f"{s}: {sizes[s]}" for s in COMPOSITE)
    excluded = rows[0][1].num_excluded_n

    if args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["model"] + columns)
        for name, result in rows:
            writer.writerow([name] + [_fmt(v) for v in result.row().values()])
        return

    if args.format == "latex":
        print(f"% images per subset: {size_line}; excluded N: {excluded}")
        print("% model & GT A\\N & & ReGT A\\N (+delta) & S & S+ & S- & M & M+ & M- \\\\")
        for name, result in rows:
            r = result.row()
            gt = _fmt(r["GT A\\N"])
            delta = r["ReGT-GT A\\N"]
            regt = [_fmt(r[f"ReGT {s}"]) for s in COMPOSITE]
            regt[0] = regt[0] + " {\\scriptsize{(" + f"{delta:+.2f}" + ")}}"
            print(f"{name} & {gt} & & " + " & ".join(regt) + " \\\\")
        return

    width = max(len(n) for n in names + ["model"])
    print(f"Images per subset: {size_line} (excluded N: {excluded})")
    print("model".ljust(width) + "".join(c.rjust(14) for c in columns))
    for name, result in rows:
        print(name.ljust(width) + "".join(_fmt(v).rjust(14) for v in result.row().values()))


if __name__ == "__main__":
    main()
