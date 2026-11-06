"""Label statistics of ReImageNet.

Reports the image categories (N, S+, S-, M+, M-), the share of images whose original
ImageNet label is not a valid ReImageNet label, and box and attribute counts.

Definitions (eq-class pairs count as one class, "not sure" is not a valid label):

- incorrect original label: the original label is not among the image's valid
  ReImageNet labels, i.e. S- + M- + N, over all images;
- multilabel: M, over all images;
- no ImageNet-1k object: N, over all images.

Usage:
    python -m eval_reimagenet.label_stats
"""

import argparse
import statistics
from pathlib import Path

from .annotations import DEFAULT_DATA_DIR, NOT_SURE, load_reimagenet
from .metrics import GROUPS, group_images

ATTRIBUTES = ["crowd_flag", "ocr_needed_flag", "rendition_flag", "reflected_flag", "dominant_object"]


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:6.2f}%" if d else "     -"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="ReImageNet release directory")
    args = parser.parse_args()

    annotations, eq, _ = load_reimagenet(args.data_dir)
    groups = group_images(annotations, eq)
    n = {g: len(groups[g]) for g in GROUPS}
    total = len(annotations)
    evaluated = total - n["N"]

    print(f"Images: {total}  (eq-class groups: {len(eq)})\n")
    print("Category                              images   of all  of A\\N")
    rows = [
        ("N   no valid ImageNet-1k class", n["N"]),
        ("S   one class", n["S+"] + n["S-"]),
        ("S+  ... original label correct", n["S+"]),
        ("S-  ... original label wrong", n["S-"]),
        ("M   more than one class", n["M+"] + n["M-"]),
        ("M+  ... original label among them", n["M+"]),
        ("M-  ... original label not among them", n["M-"]),
    ]
    for label, count in rows:
        of_eval = "       " if label.startswith("N ") else _pct(count, evaluated)
        print(f"{label:<38}{count:6d}  {_pct(count, total)}  {of_eval}")

    wrong = n["S-"] + n["M-"]
    print()
    print(f"Original label not a valid ReImageNet label (S- + M- + N): {wrong + n['N']:6d}  {_pct(wrong + n['N'], total)} of all")
    print(f"Original label wrong, images with a valid label (S- + M-): {wrong:6d}  {_pct(wrong, evaluated)} of A\\N")
    print(f"Multilabel (M):                                            {n['M+'] + n['M-']:6d}  {_pct(n['M+'] + n['M-'], total)} of all")
    print(f"No ImageNet-1k object (N):                                 {n['N']:6d}  {_pct(n['N'], total)} of all")
    not_sure_only = sum(1 for a in annotations if a.reannotated_labels == {NOT_SURE})
    print(f"  of which only 'not sure' labels:                         {not_sure_only:6d}")

    boxes = [b for a in annotations for b in a.bboxes]
    per_image_boxes = [len(a.bboxes) for a in annotations]
    per_image_classes = [eq.num_distinct(a.reannotated_labels) for a in annotations]
    print()
    print(f"Boxes: {len(boxes)}  (per image: mean {statistics.mean(per_image_boxes):.2f}, "
          f"median {statistics.median(per_image_boxes):g})")
    print(f"Valid classes per image: mean {statistics.mean(per_image_classes):.2f}, "
          f"median {statistics.median(per_image_classes):g}")
    print("\nBox attributes                 boxes  of boxes   images  of images")
    for attr in ATTRIBUTES:
        n_boxes = sum(1 for b in boxes if getattr(b, attr))
        n_images = sum(1 for a in annotations if any(getattr(b, attr) for b in a.bboxes))
        print(f"{attr:<28}{n_boxes:8d}  {_pct(n_boxes, len(boxes))}  {n_images:7d}  {_pct(n_images, total)}")


if __name__ == "__main__":
    main()
