"""Loading top-1 predictions of a model on the ImageNet-1k validation set.

Supported formats (class ids are ImageNet-1k indices 0-999, ``-1`` = no prediction):

CSV with an ``img_id`` column and a prediction column (``top_1_pred`` by default)::

    img_id,top_1_pred,...
    ILSVRC2012_val_00000001.JPEG,65,...

JSON, optionally nested under ``"predictions"``, with a class id per image::

    {"predictions": {"ILSVRC2012_val_00000001.JPEG": 65, ...}}
    {"predictions": {"ILSVRC2012_val_00000001.JPEG": {"label": 65, "similarity": 0.95}, ...}}

A list value is read as a ranked list and its first element is used. Free-text class
names must be mapped to class ids first.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Union


def _image_key(name: str) -> str:
    name = str(name).strip()
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    return name if name.upper().endswith(".JPEG") else f"{name}.JPEG"


def _to_class_id(value, image: str) -> int:
    if isinstance(value, dict):
        if "label" not in value:
            raise ValueError(f"{image}: prediction dict has no 'label' field: {value}")
        value = value["label"]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else -1
    if isinstance(value, bool):
        raise ValueError(f"{image}: invalid prediction {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(
        f"{image}: prediction {value!r} is not a class id. Map class names to ImageNet-1k ids first."
    )


def load_predictions(path: Union[str, Path], column: Optional[str] = None) -> Dict[str, int]:
    """Top-1 class id per image name (``ILSVRC2012_val_XXXXXXXX.JPEG``)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
            data = data["predictions"]
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object mapping image names to predictions")
        return {_image_key(k): _to_class_id(v, k) for k, v in data.items()}

    column = column or "top_1_pred"
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "img_id" not in fields:
            raise ValueError(f"{path}: CSV has no 'img_id' column (columns: {fields})")
        if column not in fields:
            raise ValueError(f"{path}: CSV has no '{column}' column (columns: {fields})")
        return {_image_key(row["img_id"]): _to_class_id(row[column], row["img_id"]) for row in reader}
