"""Evaluation of ImageNet-1k classifiers with ReImageNet labels.

Annotations: https://huggingface.co/datasets/vrg-prague/ReImageNet
"""

from .annotations import (
    Annotation,
    BoundingBox,
    EqClasses,
    load_annotations,
    load_eq_classes,
    load_label_names,
    load_reimagenet,
)
from .metrics import GROUPS, categorize, evaluate, group_images
from .predictions import load_predictions

__all__ = [
    "Annotation",
    "BoundingBox",
    "EqClasses",
    "GROUPS",
    "categorize",
    "evaluate",
    "group_images",
    "load_annotations",
    "load_eq_classes",
    "load_label_names",
    "load_predictions",
    "load_reimagenet",
]
