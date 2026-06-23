from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Set

project_root = Path(__file__).parent.parent.parent

from src.data_models import BoundingBox, ImageAnnotation


def _parse_hf_bbox(b: dict) -> BoundingBox:
    return BoundingBox(
        coordinates=b["coordinates"],
        labels=set(b["labels"]),
        group=b.get("group"),
        crowd_flag=bool(b.get("crowd_flag", False)),
        reflected_flag=bool(b.get("reflected_flag", False)),
        rendition_flag=bool(b.get("rendition_flag", False)),
        ocr_needed_flag=bool(b.get("ocr_needed_flag", False)),
        dominant_object=bool(b.get("dominant_object", False)),
    )


class AnnotationLoaderMixin:
    """Mixin providing annotation loading from the HuggingFace ReImageNet dataset."""

    def load_annotations(self) -> List[ImageAnnotation]:
        """
        Load all annotations from vrg-prague/ReImageNet on HuggingFace.

        Returns:
            List of ImageAnnotation objects, optionally filtered by class or image label filters.
        """
        from datasets import load_dataset

        self.logger.info("Loading annotations from HuggingFace: vrg-prague/ReImageNet")
        ds = load_dataset("vrg-prague/ReImageNet", split="test")

        annotations: List[ImageAnnotation] = []
        for record in ds:
            image_name: str = record["image_name"]
            # original_class is int[] — usually one element; take first
            original_class: int = record["original_class"][0]
            reannotated_labels: Set[int] = set(record["reannotated_labels"])
            file_path = Path(record["file_path"])

            bboxes = [_parse_hf_bbox(b) for b in (record.get("bboxes") or [])]

            annotations.append(ImageAnnotation(
                image_name=image_name,
                original_class=original_class,
                reannotated_labels=reannotated_labels,
                annotator="vrg",
                file_path=file_path,
                bboxes=bboxes,
            ))

        self.logger.info(f"Loaded {len(annotations)} annotations from HF")
        return annotations

