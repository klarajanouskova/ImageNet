from __future__ import annotations
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BoundingBox:
    """Bounding box with label(s) and quality flags.

    Boxes can have multiple labels when they share the same 'group' key in the annotation.
    The 'labels' set contains all valid class labels for this bounding box.
    """
    coordinates: List[float]
    labels: Set[int]  # All labels for this box (single label for ungrouped, multiple for grouped boxes)
    group: Optional[int] = None  # Group ID for boxes that share the same physical region
    crowd_flag: bool = False
    reflected_flag: bool = False
    rendition_flag: bool = False
    ocr_needed_flag: bool = False
    dominant_object: bool = False


@dataclass
class ImageAnnotation:
    """Data class to store image annotation information."""
    image_name: str
    original_class: int
    reannotated_labels: Set[int]
    annotator: str
    file_path: Path
    bboxes: List[BoundingBox] = field(default_factory=list)


@dataclass
class AccuracyResults:
    """Data class to store accuracy calculation results."""
    total_images: int
    group_sizes: Dict[str, int]
    accuracies: Dict[str, float]

    def __str__(self) -> str:
        lines = [
            f"Total images: {self.total_images}",
            "\nGroup sizes:",
        ]
        for group, size in self.group_sizes.items():
            lines.append(f"  {group}: {size}")

        lines.append("\nTop-1 accuracies:")
        for group, acc in self.accuracies.items():
            lines.append(f"  Acc_{group}^1: {acc:.2f}")
        acc_line = []
        for group, acc in self.accuracies.items():
            acc_line.append(f"{acc:.2f}")
        print(" & ".join(acc_line))

        return "\n".join(lines)

