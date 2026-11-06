"""Image categories and top-1 accuracy with original and ReImageNet labels.

Let ``alpha_i`` be the set of valid ReImageNet classes of image ``i`` ("not sure" removed,
an eq-class pair counted as one class) and ``y_i`` its original ImageNet label.

- ``N``: ``|alpha_i| = 0``, no valid ImageNet-1k class. Excluded from evaluation.
- ``S``: ``|alpha_i| = 1``. ``S+`` if ``y_i`` in ``alpha_i``, otherwise ``S-``.
- ``M``: ``|alpha_i| > 1``. ``M+`` if ``y_i`` in ``alpha_i``, otherwise ``M-``.
- ``A\\N``: all images except ``N``.

A top-1 prediction ``p_i`` is correct

- with the original labels (``GT``) if ``p_i`` equals ``y_i``,
- with ReImageNet labels (``ReGT``) if ``p_i`` is in ``alpha_i``,

where "equals" and "in" treat the classes of an eq-class pair as the same class.
Both are reported on ``A\\N``, so they use the same images.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .annotations import Annotation, EqClasses

GROUPS = ["N", "S+", "S-", "M+", "M-"]
"""Disjoint categories. ``S``, ``M`` and ``A\\N`` are unions of them."""

COMPOSITE = OrderedDict(
    [
        ("A\\N", ["S+", "S-", "M+", "M-"]),
        ("S", ["S+", "S-"]),
        ("S+", ["S+"]),
        ("S-", ["S-"]),
        ("M", ["M+", "M-"]),
        ("M+", ["M+"]),
        ("M-", ["M-"]),
    ]
)
"""Evaluated subsets, in the column order of the paper's main table."""


def categorize(annotation: Annotation, eq: EqClasses) -> str:
    """Category of one image: one of ``N``, ``S+``, ``S-``, ``M+``, ``M-``."""
    num = eq.num_distinct(annotation.reannotated_labels)
    if num == 0:
        return "N"
    sign = "+" if eq.matches_any(annotation.original_class, annotation.valid_labels) else "-"
    return ("S" if num == 1 else "M") + sign


def group_images(annotations: List[Annotation], eq: EqClasses) -> Dict[str, List[Annotation]]:
    """Split annotations into the disjoint categories in ``GROUPS``."""
    groups: Dict[str, List[Annotation]] = {g: [] for g in GROUPS}
    for annotation in annotations:
        groups[categorize(annotation, eq)].append(annotation)
    return groups


@dataclass
class EvalResult:
    """Counts of evaluated and correct images per subset.

    ``total[k]`` and ``correct_regt[k]`` are defined for every key of ``COMPOSITE``.
    ``correct_gt`` is defined for the same keys, with the original labels.
    """

    total: Dict[str, int] = field(default_factory=dict)
    correct_gt: Dict[str, int] = field(default_factory=dict)
    correct_regt: Dict[str, int] = field(default_factory=dict)
    num_excluded_n: int = 0
    num_missing: int = 0

    @staticmethod
    def _pct(correct: int, total: int) -> float:
        return round(100.0 * correct / total, 2) if total else float("nan")

    def acc_gt(self, subset: str = "A\\N") -> float:
        return self._pct(self.correct_gt[subset], self.total[subset])

    def acc_regt(self, subset: str = "A\\N") -> float:
        return self._pct(self.correct_regt[subset], self.total[subset])

    def row(self) -> "OrderedDict[str, float]":
        """Values of one row of the paper's main table."""
        row = OrderedDict()
        row["GT A\\N"] = self.acc_gt("A\\N")
        for subset in COMPOSITE:
            row[f"ReGT {subset}"] = self.acc_regt(subset)
        row["ReGT-GT A\\N"] = round(row["ReGT A\\N"] - row["GT A\\N"], 2)
        return row


def evaluate(
    annotations: List[Annotation],
    predictions: Mapping[str, int],
    eq: EqClasses,
    allow_missing: bool = False,
) -> EvalResult:
    """Top-1 accuracy of ``predictions`` (image name -> class id) per image category.

    Images in ``N`` are excluded. Every other image must have a prediction, unless
    ``allow_missing`` is set, in which case images without a prediction are skipped
    and their number is stored in ``num_missing``.
    """
    groups = group_images(annotations, eq)
    result = EvalResult(num_excluded_n=len(groups["N"]))

    per_group_total: Dict[str, int] = {}
    per_group_gt: Dict[str, int] = {}
    per_group_regt: Dict[str, int] = {}
    missing: List[str] = []

    for group in GROUPS:
        if group == "N":
            continue
        total = gt = regt = 0
        for annotation in groups[group]:
            pred: Optional[int] = predictions.get(annotation.image_name)
            if pred is None:
                missing.append(annotation.image_name)
                continue
            total += 1
            gt += eq.same(pred, annotation.original_class) and pred != -1
            regt += eq.matches_any(pred, annotation.valid_labels)
        per_group_total[group] = total
        per_group_gt[group] = gt
        per_group_regt[group] = regt

    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)} evaluated images have no prediction (e.g. {missing[:3]}). "
            "Pass allow_missing=True (--allow-missing) to evaluate only the predicted images."
        )
    result.num_missing = len(missing)

    for subset, parts in COMPOSITE.items():
        result.total[subset] = sum(per_group_total[p] for p in parts)
        result.correct_gt[subset] = sum(per_group_gt[p] for p in parts)
        result.correct_regt[subset] = sum(per_group_regt[p] for p in parts)
    return result
