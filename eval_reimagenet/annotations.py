"""Loading the ReImageNet release files.

The release (https://huggingface.co/datasets/vrg-prague/ReImageNet) has three files:

- ``reannotation.jsonl``: one record per ImageNet-1k validation image.
- ``label_names.json``: the 1000 synset ids, indexed by class id (0-999).
- ``class_update_config.json``: pairs of equivalent classes ("eq_classes"), i.e.
  ImageNet classes that cannot be told apart visually. A prediction of either class
  of a pair is accepted for the other one, both for the original and the ReImageNet labels.

Class id ``-1`` in ``reannotated_labels`` or in a box means "not sure". It is never
a valid label.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

NOT_SURE = -1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("REIMAGENET_DIR", REPO_ROOT / "data" / "reimagenet"))

ANNOTATIONS_FILE = "reannotation.jsonl"
LABEL_NAMES_FILE = "label_names.json"
CLASS_CONFIG_FILE = "class_update_config.json"


@dataclass
class BoundingBox:
    """A box with its class labels and attributes.

    Coordinates are ``[x1, y1, x2, y2]`` in pixels of the original image.
    A box has more than one label when annotators grouped several labels on one region.
    """

    coordinates: List[float]
    labels: Set[int]
    group: Optional[int] = None
    crowd_flag: bool = False
    reflected_flag: bool = False
    rendition_flag: bool = False
    ocr_needed_flag: bool = False
    dominant_object: bool = False


@dataclass
class Annotation:
    """ReImageNet annotation of one validation image."""

    image_name: str
    file_path: str
    original_class: int
    """Original ImageNet label, i.e. the class of the image's synset folder."""
    original_classes: List[int]
    """Original label(s) as stored in the release. Two ids only for an eq-class pair."""
    reannotated_labels: Set[int]
    """All labels of the image, may contain ``-1`` (not sure)."""
    bboxes: List[BoundingBox] = field(default_factory=list)

    @property
    def valid_labels(self) -> Set[int]:
        """Reannotated labels without "not sure"."""
        return self.reannotated_labels - {NOT_SURE}


class EqClasses:
    """Equivalent ImageNet classes from ``class_update_config.json``.

    Every class is mapped to a canonical id (the smallest id of its group), so two
    classes are equivalent iff they have the same canonical id.
    """

    def __init__(self, groups: Union[Dict[str, List[int]], Iterable[Iterable[int]]] = ()):
        if isinstance(groups, dict):
            groups = groups.values()
        self.groups: List[Set[int]] = [set(g) for g in groups if g]
        self._canonical: Dict[int, int] = {}
        for group in self.groups:
            rep = min(group)
            for c in group:
                self._canonical[c] = rep

    def canonical(self, class_id: int) -> int:
        return self._canonical.get(class_id, class_id)

    def same(self, a: int, b: int) -> bool:
        return self.canonical(a) == self.canonical(b)

    def matches_any(self, class_id: int, labels: Iterable[int]) -> bool:
        """True if ``class_id`` is (equivalent to) any of ``labels``. ``-1`` never matches."""
        if class_id == NOT_SURE:
            return False
        c = self.canonical(class_id)
        return any(label != NOT_SURE and self.canonical(label) == c for label in labels)

    def num_distinct(self, labels: Iterable[int]) -> int:
        """Number of distinct valid classes; an eq-class pair counts as one class."""
        return len({self.canonical(label) for label in labels if label != NOT_SURE})

    def __len__(self) -> int:
        return len(self.groups)


def load_label_names(path: Union[str, Path]) -> List[str]:
    """Synset ids indexed by class id."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eq_classes(path: Union[str, Path]) -> EqClasses:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return EqClasses(config.get("eq_classes", {}))


def load_annotations(path: Union[str, Path], label_names: Optional[List[str]] = None) -> List[Annotation]:
    """Load ``reannotation.jsonl``.

    Boxes keep the file order. ``original_class`` is the class of the image's synset
    folder (``file_path``), which is the original ImageNet label. When the release lists
    two original classes, they are an eq-class pair, so either one gives the same result.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ReImageNet annotations not found: {path}. "
            "Run `python -m eval_reimagenet.download` first, or set REIMAGENET_DIR."
        )
    synset_to_id = {s: i for i, s in enumerate(label_names)} if label_names else {}

    annotations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            originals = [int(c) for c in rec["original_class"]]
            folder_class = synset_to_id.get(rec["file_path"].split("/")[0])
            original_class = folder_class if folder_class in originals else originals[0]

            bboxes = [
                BoundingBox(
                    coordinates=list(b["coordinates"]),
                    labels={int(label) for label in b["labels"]},
                    group=b.get("group"),
                    crowd_flag=bool(b.get("crowd_flag", False)),
                    reflected_flag=bool(b.get("reflected_flag", False)),
                    rendition_flag=bool(b.get("rendition_flag", False)),
                    ocr_needed_flag=bool(b.get("ocr_needed_flag", False)),
                    dominant_object=bool(b.get("dominant_object", False)),
                )
                for b in rec.get("bboxes", [])
            ]
            annotations.append(
                Annotation(
                    image_name=rec["image_name"],
                    file_path=rec["file_path"],
                    original_class=original_class,
                    original_classes=originals,
                    reannotated_labels={int(label) for label in rec["reannotated_labels"]},
                    bboxes=bboxes,
                )
            )
    return annotations


def load_reimagenet(data_dir: Union[str, Path, None] = None) -> Tuple[List[Annotation], EqClasses, List[str]]:
    """Load annotations, equivalent classes and label names from a release directory."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    label_names = load_label_names(data_dir / LABEL_NAMES_FILE)
    eq_classes = load_eq_classes(data_dir / CLASS_CONFIG_FILE)
    annotations = load_annotations(data_dir / ANNOTATIONS_FILE, label_names)
    return annotations, eq_classes, label_names
