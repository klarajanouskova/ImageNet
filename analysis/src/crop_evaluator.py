from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MAPPINGS_DIR
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent

from src.data_models import BoundingBox, ImageAnnotation, AccuracyResults


class CropEvaluatorMixin:
    """Mixin providing crop-based accuracy evaluation for AccuracyCalculator."""

    @staticmethod
    def _get_pred_column(df: pd.DataFrame) -> str:
        if "k_11_pred" in df.columns:
            return "k_11_pred"
        return "top_1_pred"

    def compute_all_crops_accuracy(
        self,
        csv_path: Path,
        class_lookup: Union[List | None] = None,
        no_fill: bool = False,
    ) -> Tuple[float, int, int]:
        self.class_lookup = class_lookup
        df = pd.read_csv(csv_path)

        img_names = (
            df["img_id"]
            .str.split("_crop_", n=1)
            .str[0]
            .add(".JPEG")
            .tolist()
        )

        pred_col = self._get_pred_column(df)
        regt_labels = df["original_label"].tolist()
        pred_labels = df[pred_col].tolist()

        acc_wrt_regt_list = [self.is_correct(regt, pred) for regt, pred in zip(regt_labels, pred_labels)]

        if not no_fill:
            imgs_in_file = set(img_names)
            all_val_imgs = {f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)}
            num_missing = len(all_val_imgs - imgs_in_file)
            acc_wrt_regt_list.extend([True] * num_missing)
            print(f"Missing images (treated correct for regt): {num_missing}")

        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        acc = n_correct / n_total * 100 if n_total > 0 else 0.0
        return acc, n_correct, n_total

    def compute_crop_oracle_from_csv(
        self,
        csv_path: Path,
        class_lookup: Union[List | None] = None,
    ) -> Tuple[float, int, int]:
        """Oracle accuracy over annotated images: an image is correct if ANY crop is correct."""
        self.class_lookup = class_lookup
        df = pd.read_csv(csv_path)

        df["img_name"] = df["img_id"].str.split("_crop_", n=1).str[0] + ".JPEG"

        all_val_imgs = {f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)}
        imgs_in_file = set(df["img_name"].unique())
        num_missing = len(all_val_imgs - imgs_in_file)

        pred_col = self._get_pred_column(df)
        acc_wrt_regt_list = [
            any(self.is_correct(regt, pred) for regt, pred in zip(group["original_label"].tolist(), group[pred_col].tolist()))
            for _, group in df.groupby("img_name")
        ]
        acc_wrt_regt_list.extend([True] * num_missing)

        print(f"Images with crops: {len(imgs_in_file)} | Missing images (treated correct for regt): {num_missing}")
        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        return n_correct / n_total * 100 if n_total > 0 else 0.0, n_correct, n_total

    def compute_largest_crop_accuracy(
        self,
        csv_path: Path,
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Accuracy for 50k val images using only the largest crop per image.

        Mapping loaded from data/mappings/image_to_largest_crop.json.
        Images with no mapping entry or whose crop is absent from the CSV are
        treated as correct for regt.
        """
        mapping_path = MAPPINGS_DIR / "image_to_largest_crop.json"
        if not mapping_path.exists():
            logging.getLogger(__name__).info("image_to_largest_crop.json not found — skipping largest-crop accuracy")
            return None, None, None

        with open(mapping_path) as f:
            img_to_crop: dict = json.load(f)

        crop_df = pd.read_csv(csv_path).set_index("img_id")
        crop_pred_col = self._get_pred_column(crop_df)

        acc_wrt_regt_list = []
        for img_name in (f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)):
            crop_id = img_to_crop.get(img_name)
            if crop_id is not None and crop_id in crop_df.index:
                pred = int(crop_df.at[crop_id, crop_pred_col])
                regt = int(crop_df.at[crop_id, "original_label"])
                acc_wrt_regt_list.append(self.is_correct(regt, pred))
            else:
                acc_wrt_regt_list.append(True)

        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        return n_correct / n_total * 100 if n_total > 0 else 0.0, n_correct, n_total

    # ------------------------------------------------------------------
    # LLM-JSON variants of the crop accuracy methods.
    # Crop JSON format: {"predictions": {"<img>_crop_<regt>_<idx>": {"label": int, ...}, ...}}
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_crop_key(crop_key: str):
        """Return (img_name_with_jpeg, regt_label) from a crop key like 'ILSVRC2012_val_00000001_crop_65_0'."""
        base, rest = crop_key.rsplit("_crop_", 1)
        regt = int(rest.split("_")[0])
        return base + ".JPEG", regt

    def compute_all_crops_accuracy_from_json(
        self,
        json_path: Path,
        class_lookup=None,
        no_fill: bool = False,
    ) -> Tuple[float, int, int]:
        self.class_lookup = class_lookup
        with open(json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)["predictions"]

        img_names, regt_labels, pred_labels = [], [], []
        for crop_key, entry in predictions.items():
            img_name, regt = self._parse_crop_key(crop_key)
            img_names.append(img_name)
            regt_labels.append(regt)
            pred_labels.append(int(entry["label"]))

        acc_wrt_regt_list = [self.is_correct(regt, pred) for regt, pred in zip(regt_labels, pred_labels)]

        if not no_fill:
            imgs_in_file = set(img_names)
            all_val_imgs = {f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)}
            num_missing = len(all_val_imgs - imgs_in_file)
            acc_wrt_regt_list.extend([True] * num_missing)
            print(f"Missing images (treated correct for regt): {num_missing}")

        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        return n_correct / n_total * 100 if n_total > 0 else 0.0, n_correct, n_total

    def compute_crop_oracle_from_json(
        self,
        json_path: Path,
        class_lookup=None,
    ) -> Tuple[float, int, int]:
        self.class_lookup = class_lookup
        with open(json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)["predictions"]

        groups: dict = {}
        for crop_key, entry in predictions.items():
            img_name, regt = self._parse_crop_key(crop_key)
            groups.setdefault(img_name, []).append((regt, int(entry["label"])))

        all_val_imgs = {f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)}
        imgs_in_file = set(groups.keys())
        num_missing = len(all_val_imgs - imgs_in_file)

        acc_wrt_regt_list = [
            any(self.is_correct(regt, pred) for regt, pred in entries)
            for entries in groups.values()
        ]
        acc_wrt_regt_list.extend([True] * num_missing)

        print(f"Images with crops: {len(imgs_in_file)} | Missing images (treated correct for regt): {num_missing}")
        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        return n_correct / n_total * 100 if n_total > 0 else 0.0, n_correct, n_total

    def compute_largest_crop_accuracy_from_json(
        self,
        json_path: Path,
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        mapping_path = MAPPINGS_DIR / "image_to_largest_crop.json"
        if not mapping_path.exists():
            logging.getLogger(__name__).info("image_to_largest_crop.json not found — skipping largest-crop accuracy")
            return None, None, None

        with open(mapping_path) as f:
            img_to_crop: dict = json.load(f)

        with open(json_path, "r", encoding="utf-8") as f:
            crop_preds = json.load(f)["predictions"]

        acc_wrt_regt_list = []
        for img_name in (f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)):
            crop_id = img_to_crop.get(img_name)
            if crop_id is not None and crop_id in crop_preds:
                _, regt = self._parse_crop_key(crop_id)
                acc_wrt_regt_list.append(self.is_correct(regt, int(crop_preds[crop_id]["label"])))
            else:
                acc_wrt_regt_list.append(True)

        n_correct = sum(acc_wrt_regt_list)
        n_total = len(acc_wrt_regt_list)
        return n_correct / n_total * 100 if n_total > 0 else 0.0, n_correct, n_total
