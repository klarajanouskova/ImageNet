"""
Accuracy calculator for reannotation experiments.

This module calculates Top-1 accuracy based on reannotated ImageNet labels
and categorizes images according to the mathematical formulation in the paper.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd
import numpy as np
import imagenet_classes
import matplotlib.pyplot as plt
from scipy.stats import t
from collections import defaultdict
from PIL import Image

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['text.usetex'] = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))






# Data models and mixin classes extracted into focused modules
from src.data_models import BoundingBox, ImageAnnotation, AccuracyResults
from src.annotation_loader import AnnotationLoaderMixin
from src.crop_evaluator import CropEvaluatorMixin
from src.stats_analyzer import StatsAnalyzerMixin, _plot_alluvial_corrections


class AccuracyCalculator(
    AnnotationLoaderMixin,
    CropEvaluatorMixin,
    StatsAnalyzerMixin,
):
    """
    Calculator for Top-1 accuracy based on reannotated ImageNet labels.
    
    Categorizes images into groups:
    - N: images with none of the ImageNet labels (|α_i| = 0)
    - S: images with exactly one ImageNet label (|α_i| = 1)
      - S+: original label is correct
      - S-: original label is incorrect
    - M: images with more than one ImageNet label (|α_i| > 1)
      - M+: original label is in α_i
      - M-: original label is not in α_i
    """
    
    def __init__(self, processed_data_dir: Path, random_seed: int = 42):
        """
        Initialize the accuracy calculator.

        Args:
            processed_data_dir: Path to the processed data directory
            random_seed: Random seed for reproducible results
        """
        self.logger = logging.getLogger(__name__)
        self.processed_data_dir = Path(processed_data_dir)
        self.class_lookup = None
        self.random_seed = random_seed

        self.groups = None
        self.llm_image_subset: Set[str] = set()  # Store image IDs from LLM JSON
        self._class_lookup_for_categorization = None  # Store class lookup for use in categorize_images

        # Set random seed for reproducible results
        random.seed(self.random_seed)
        self.logger.info(f"Random seed set to: {self.random_seed}")

        
    def categorize_images(self, annotations: List[ImageAnnotation]) -> Dict[str, List[ImageAnnotation]]:
        """
        Categorize images into groups based on reannotated labels.
        
        Considers equivalent classes from class_update_config.json when determining
        if the original class is in the reannotated labels.
        
        Args:
            annotations: List of image annotations
        
        Returns:
            Dictionary with group names as keys and lists of annotations as values
        """
        groups = {
            "N": [],    # |α_i| = 0
            "S": [],    # |α_i| = 1
            "S+": [],   # |α_i| = 1 and original label is correct
            "S-": [],   # |α_i| = 1 and original label is incorrect
            "M": [],    # |α_i| > 1
            "M+": [],   # |α_i| > 1 and original label is in α_i
            "M-": [],   # |α_i| > 1 and original label is not in α_i
        }
        
        print("Categorizing images into groups...")
        with tqdm(annotations, desc="Categorizing images", unit="images", disable=False) as pbar:
            for annotation in pbar:
                num_labels = len(annotation.reannotated_labels)
                # Check if any reannotated label is equivalent to the original class
                original_in_reannotated = self._is_original_in_reannotated(annotation.original_class, annotation.reannotated_labels)

                # Count effective distinct semantic labels (excluding -1, collapsing equivalent classes)
                valid_labels = {l for l in annotation.reannotated_labels if l != -1}
                if self.class_lookup is not None:
                    seen: Set[int] = set()
                    num_effective = 0
                    for lbl in valid_labels:
                        if lbl not in seen:
                            num_effective += 1
                            seen.add(lbl)
                            seen.update(self.class_lookup.get_neighbors(lbl))
                else:
                    num_effective = len(valid_labels)

                if num_labels == 0:
                    # N group: no reannotated labels
                    groups["N"].append(annotation)
                    category = "N"
                elif num_labels == 1:
                    # Check if the single label is -1 (not sure box)
                    if -1 in annotation.reannotated_labels:
                        # Treat as N group (no valid labels)
                        groups["N"].append(annotation)
                        category = "N"
                    else:
                        # S group: exactly one reannotated label
                        groups["S"].append(annotation)
                        if original_in_reannotated:
                            groups["S+"].append(annotation)
                            category = "S+"
                        else:
                            groups["S-"].append(annotation)
                            category = "S-"
                else:
                    # M group: more than one reannotated label
                    # Check if all labels are -1 (only not sure boxes)
                    if annotation.reannotated_labels == {-1}:
                        # Treat as N group (no valid labels)
                        groups["N"].append(annotation)
                        category = "N"
                    elif num_effective <= 1:
                        # All valid labels are in the same equivalent group — treat as S
                        groups["S"].append(annotation)
                        if original_in_reannotated:
                            groups["S+"].append(annotation)
                            category = "S+"
                        else:
                            groups["S-"].append(annotation)
                            category = "S-"
                    else:
                        # Uncomment the following block in case of camera ready
                        if -1 in annotation.reannotated_labels and num_labels == 2:
                            # If there are only two labels and one is -1, treat as S group
                            groups["S"].append(annotation)
                            if original_in_reannotated:
                                groups["S+"].append(annotation)
                                category = "S+"
                            else:
                                groups["S-"].append(annotation)
                                category = "S-"
                            pbar.set_postfix_str(f"Last: {category} (N:{len(groups['N'])}, S:{len(groups['S'])}, M:{len(groups['M'])})")
                            continue

                        groups["M"].append(annotation)
                        if original_in_reannotated:
                            groups["M+"].append(annotation)
                            category = "M+"
                        else:
                            groups["M-"].append(annotation)
                            category = "M-"
                
                # Update progress bar with current categorization
                pbar.set_postfix_str(f"Last: {category} (N:{len(groups['N'])}, S:{len(groups['S'])}, M:{len(groups['M'])})")
        
        return groups

    def _is_original_in_reannotated(self, original_class: int, reannotated_labels: Set[int]) -> bool:
        """
        Check if the original class matches any reannotated label, considering equivalent classes.
        
        Args:
            original_class: The original ground truth class
            reannotated_labels: Set of reannotated class labels
            
        Returns:
            True if original class is equivalent to any reannotated label
        """
        for reannotated_class in reannotated_labels:
            if reannotated_class == -1:  # Skip "not sure" label
                continue
            if self.is_correct(original_class, reannotated_class):
                return True
        return False

    def get_group(self, group_name: str) -> List[ImageAnnotation]:
        if self.groups is None:
            # Load annotations
            annotations = self.load_annotations()
            # Categorize images
            self.groups = self.categorize_images(annotations)

        if group_name in self.groups and self.groups[group_name]:
            return self.groups[group_name]
        else:
            return []

    def calculate_top1_accuracy_vs_gt(self, annotations: List[ImageAnnotation]) -> float:
        """
        Calculate Top-1 accuracy against ground truth labels for a list of annotations.
        
        Top-1 accuracy for image i:
        Acc_i^1 = 1 if L_i^1 ∈ α_i or |α_i| = 0, else 0
        
        Uses the original class as the "predicted" class for accuracy calculation
        against reannotated ground truth labels.
        
        Args:
            annotations: List of image annotations
        
        Returns:
            Top-1 accuracy as a float between 0 and 1
        """
        if not annotations:
            return 0.0
        
        correct = 0
        for annotation in annotations:
            membership_match = self.is_correct(annotation.reannotated_labels, annotation.original_class)
            # Correct or OOD image or Not Sure image
            if membership_match or len(annotation.reannotated_labels) == 0 or (len(annotation.reannotated_labels) == 1 and list(annotation.reannotated_labels)[0] == -1):
                correct += 1

        return np.round((correct / len(annotations) )* 100, 2)

    def load_csv_predictions(self, csv_path: Path, pred_column: str = 'top_1_pred') -> Dict[str, int]:
        """
        Load predictions from CSV file.
        
        Args:
            csv_path: Path to the CSV file with predictions
        
        Returns:
            Dictionary mapping image names to predicted class IDs
        """
        self.logger.info(f"Loading predictions from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            if 'img_id' not in df.columns:
                raise ValueError("CSV must contain 'img_id' column")
            if pred_column not in df.columns:
                raise ValueError(f"CSV must contain '{pred_column}' column")

            # Create mapping from image name to predicted class
            predictions = {}
            for _, row in df.iterrows():
                img_id = row['img_id']
                top_1_pred = int(row[pred_column])
                predictions[img_id] = top_1_pred
            
            self.logger.info(f"Loaded {len(predictions)} predictions from CSV")
            return predictions

        except Exception as e:
            self.logger.error(f"Error loading CSV predictions: {e}")
            raise

    def load_csv_predictions_topk(self, csv_path: Path, k: int = 5) -> Dict[str, List[int]]:
        """
        Load top-k predictions from CSV file.

        Args:
            csv_path: Path to the CSV file with predictions
            k: Number of top predictions to load (default 5)

        Returns:
            Dictionary mapping image names to list of predicted class IDs
        """
        self.logger.info(f"Loading top-{k} predictions from CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)

            # Validate required columns
            if 'img_id' not in df.columns:
                raise ValueError("CSV must contain 'img_id' column")

            # Find available prediction columns
            pred_columns = []
            for i in range(1, k + 1):
                col_name = f'top_{i}_pred'
                if col_name in df.columns:
                    pred_columns.append(col_name)
                else:
                    break  # Stop if column doesn't exist

            if not pred_columns:
                raise ValueError("CSV must contain at least 'top_1_pred' column")

            self.logger.info(f"Found {len(pred_columns)} prediction columns: {pred_columns}")

            # Create mapping from image name to list of predicted classes
            predictions = {}
            for _, row in df.iterrows():
                img_id = row['img_id']
                preds = [int(row[col]) for col in pred_columns]
                predictions[img_id] = preds

            self.logger.info(f"Loaded {len(predictions)} images with top-{len(pred_columns)} predictions each")
            return predictions

        except Exception as e:
            self.logger.error(f"Error loading CSV predictions: {e}")
            raise
    
    def calculate_top1_accuracy_vs_csv(self, annotations: List[ImageAnnotation],
                                       csv_predictions: Dict[str, List[int]],
                                       evaluate_intersection_only: bool = False,
                                       export_file: Union[str, None] = None,
                                       filter_by_llm_subset: bool = False,
                                       return_counts: bool = False,
                                       confidences: Dict[str, float] = None) -> Union[float, Tuple[float, int, int],  Tuple[float, int, int, float], Tuple[float, float]]:
        """
        Calculate Top-1 accuracy (percentage) against CSV predictions.

        Acc_i^1 = 1 if predicted_class ∈ α_i or |α_i| = 0, else 0

        Behavior is controlled by evaluate_intersection_only:
        - True: Evaluate only the intersection of image IDs present in both annotations and csv_predictions.
        - False (default): Automatically choose driver list for convenience (CSV if smaller, else annotations),
          skipping items that don't have a counterpart.

        If export_file is not None, generates a JSON file with img ids, predicted labels, and annotations.

        Args:
            annotations: List of image annotations (defines the evaluable set)
            csv_predictions: Dict[img_id -> predicted class id]
            evaluate_intersection_only: If True, restrict evaluation strictly to intersection.
            filter_by_llm_subset: If True, only evaluate images in self.llm_image_subset
            export_file: If not None, export a JSON file with evaluation details. Used as a file postfix.
            return_counts: If True, return tuple (accuracy, correct, total) instead of just accuracy

        Returns:
            If return_counts=False: Top-1 accuracy in percent (0-100, rounded to 2 decimals)
            If return_counts=True: Tuple of (accuracy_percent, num_correct, num_evaluated)
        """
        if not annotations and not csv_predictions:
            return 0.0, None

        # Fast lookup from image name to annotation
        ann_by_name = {a.image_name: a for a in annotations}

        # Filter by LLM subset if requested
        if filter_by_llm_subset and self.llm_image_subset:
            ann_by_name = {k: v for k, v in ann_by_name.items() if k in self.llm_image_subset}
            csv_predictions = {k: v for k, v in csv_predictions.items() if k in self.llm_image_subset}

        correct = 0
        total_evaluated = 0
        avg_confidence = None

        if evaluate_intersection_only:
            # Strict intersection of keys
            common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())
            if confidences and len(confidences) > 0:
                avg_confidence = sum(confidences[img_id] for img_id in common_ids if img_id in confidences) / len(common_ids)

            for img_id in common_ids:
                ann = ann_by_name[img_id]
                predicted_ids = csv_predictions[img_id]
                total_evaluated += 1
                for predicted_class in predicted_ids:

                    membership_match = self.is_correct(ann.reannotated_labels, predicted_class)
                    # Correct or OOD image or Not Sure image
                    if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                        correct += 1
                        break
            denom = len(common_ids)
            basis = "intersection"
        else:
            use_csv_as_driver = len(csv_predictions) < len(ann_by_name)

            if use_csv_as_driver:
                conf_sum = 0.0
                # Iterate over CSV predictions, evaluate intersection with available annotations
                for img_id, predicted_ids in csv_predictions.items():
                    if confidences and len(confidences) > 0:
                        conf_sum += confidences.get(img_id, 0.0)
                        
                    ann = ann_by_name.get(img_id)
                    if ann is None:
                        self.logger.warning(f"No annotation found for CSV image: {img_id}")
                        continue

                    total_evaluated += 1
                    for predicted_class in predicted_ids:
                        membership_match = self.is_correct(ann.reannotated_labels, predicted_class)
                        # Correct or OOD image or Not Sure image
                        if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                            correct += 1
                            break

                denom = len(csv_predictions)
                if confidences and len(confidences) > 0 and total_evaluated > 0:
                    avg_confidence = conf_sum / total_evaluated
                basis = "CSV"
            else:
                # Iterate over annotations, evaluate those that have CSV predictions
                for ann in annotations:
                    if filter_by_llm_subset and self.llm_image_subset and ann.image_name not in self.llm_image_subset:
                        continue

                    predicted_ids = csv_predictions.get(ann.image_name)
                    if predicted_ids is None:
                        self.logger.warning(f"No prediction found for image: {ann.image_name}")
                        continue

                    total_evaluated += 1
                    # Handle both list of predictions (LLM) and single prediction (CSV)
                    if isinstance(predicted_ids, list):
                        for predicted_class in predicted_ids:
                            membership_match = self.is_correct(ann.reannotated_labels, predicted_class)
                            # Correct or OOD image or Not Sure image
                            if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                                correct += 1
                                break
                    else:
                        # Single prediction (regular CSV)
                        membership_match = self.is_correct(ann.reannotated_labels, predicted_ids)
                        # Correct or OOD image or Not Sure image
                        if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                            correct += 1

                denom = len(ann_by_name) if filter_by_llm_subset else len(annotations)
                basis = "annotations"

        if total_evaluated == 0:
            self.logger.warning("No images could be evaluated - no matching annotations/predictions found")
            if return_counts:
                return 0.0, 0, 0
            return 0.0

        accuracy = correct / total_evaluated
        subset_note = " (LLM subset)" if filter_by_llm_subset and self.llm_image_subset else ""
        self.logger.info(
            f"Evaluated {total_evaluated}/{denom} images against CSV predictions (based on {basis}){subset_note}"
        )
        
        if avg_confidence:
            self.logger.info(f"Average confidence for evaluated images: {avg_confidence:.4f}")

        if export_file:
            self._export_evaluation_details2json(annotations, csv_predictions, evaluate_intersection_only, export_file)

        accuracy_percent = np.round(accuracy * 100, 2)
        if return_counts:
            if avg_confidence:
                return accuracy_percent, correct, total_evaluated, avg_confidence
            else:
                return accuracy_percent, correct, total_evaluated
        elif avg_confidence:
            return accuracy_percent, avg_confidence
        return accuracy_percent

    def _export_evaluation_details2json(self, annotations: List[ImageAnnotation], csv_predictions: Dict[str, int], evaluate_intersection_only: bool, export_file: str):
        """
        Export evaluation details to a JSON file.

        Args:
            annotations: List of image annotations.
            csv_predictions: Dictionary of image predictions.
            evaluate_intersection_only: Whether to evaluate only the intersection of annotations and predictions.
            export_file: String to use as a postfix for the output file name.
        """
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys()) if evaluate_intersection_only else ann_by_name.keys()

        evaluation_details = {}

        for img_id in common_ids:
            ann = ann_by_name[img_id]
            predicted_ids = csv_predictions.get(img_id, "N/A")

            evaluation_details[img_id] = {
                "predicted_class": predicted_ids,
                "annotation_labels": list(ann.reannotated_labels)
            }

        output_path = self.processed_data_dir / "evaluation_reports" / f"evaluation_report_{export_file}_{len(common_ids)}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_details, f, indent=4)

    def calculate_all_accuracies_vs_gt(self, class_lookup: Union[List | None]) -> AccuracyResults:
        """
        Calculate accuracies for all groups against ground truth labels.
        
        Returns:
            AccuracyResults object containing all results
        """
        # Load annotations
        annotations = self.load_annotations()

        # Set lookup table
        self.class_lookup = class_lookup
        
        # Categorize images
        groups = self.categorize_images(annotations)
        
        # Calculate group sizes
        group_sizes = {group: len(annotations) for group, annotations in groups.items()}
        
        # Calculate accuracies for each group
        accuracies = {}
        
        # Overall accuracy (A group)
        accuracies["A"] = self.calculate_top1_accuracy_vs_gt(annotations)
        
        # Only calculate accuracies for main groups (S, M), not subgroups
        group_names = ["S", "M"]
        for group_name in group_names:
            if group_name in groups and groups[group_name]:  # Only calculate if group exists and is not empty
                accuracies[group_name] = self.calculate_top1_accuracy_vs_gt(groups[group_name])
            else:
                accuracies[group_name] = 0.0
        
        # Log detailed information
        self.logger.info("Image categorization complete:")
        for group_name, size in group_sizes.items():
            self.logger.info(f"  {group_name}: {size} images")

        return AccuracyResults(
            total_images=len(annotations),
            group_sizes=group_sizes,
            accuracies=accuracies,
        )

    def calculate_all_accuracies_vs_csv(self, csv_path: Path, class_lookup: Union[List | None], use_knn: bool = False, llm_subset: bool = False, multilabel_stats: bool = False, flag_stats: bool = False, biggest_bbox_stats: str = None, image_base_path: str = None) -> AccuracyResults:
        """
        Calculate accuracies for all groups against CSV predictions.
        
        Args:
            csv_path: Path to the CSV file with predictions
            class_lookup: Optional class lookup for equivalence groups
            use_knn: If True, use KNN predictions for evaluation
            multilabel_stats: If True, print accuracy stats by unique label count and return None
            flag_stats: If True, print accuracy stats by bbox flags for S category and return None
            biggest_bbox_stats: If set, print accuracy stats considering k largest bboxes as valid GT.
                               "cumulative" = k largest bboxes, "single" = only k-th largest bbox
            image_base_path: Base path to ImageNet validation images (for biggest_bbox_stats)

        Returns:
            AccuracyResults object containing all results (or None if multilabel_stats=True or flag_stats=True or biggest_bbox_stats is set)
        """
        # Load annotations
        annotations = self.load_annotations()
        
        # Set lookup table
        self.class_lookup = class_lookup
        
        # Categorize images
        groups = self.categorize_images(annotations)
        
        # Calculate group sizes
        group_sizes = {group: len(annotations) for group, annotations in groups.items()}

        if use_knn:
            k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]
            results = []
            
            for k in k_values:
                csv_predictions = self.load_csv_predictions(csv_path, pred_column=f"k_{k}_pred")
                
                # If multilabel_stats mode is enabled, calculate and print stats then return
                if multilabel_stats:
                    print(f"\n" + "="*80)
                    print(f"MULTILABEL STATISTICS (k={k}) - Accuracy by unique label count")
                    print("="*80)
                    # Convert single predictions to list format
                    csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
                    self.calculate_multilabel_stats(annotations, csv_like_preds)
                    continue
                
                # If flag_stats mode is enabled, calculate and print flag stats for S category
                if flag_stats:
                    print(f"\n" + "="*80)
                    print(f"FLAG STATISTICS (k={k}) - Accuracy for S category by bbox flags")
                    print("="*80)
                    s_annotations = groups.get("S", [])
                    csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
                    self.calculate_flag_stats(s_annotations, csv_like_preds)
                    continue

                # If biggest_bbox_stats mode is enabled, calculate and print biggest bbox stats
                if biggest_bbox_stats:
                    print(f"\n" + "="*80)
                    print(f"BIGGEST BBOX STATISTICS (knn k={k})")
                    print("="*80)
                    csv_like_preds = {img_id: [v] for img_id, v in csv_predictions.items()}
                    self.calculate_biggest_bbox_stats(annotations, csv_like_preds, image_base_path, mode=biggest_bbox_stats)
                    continue

                # Calculate accuracies for each group
                accuracies = {}
                
                # Calculate accuracy vs ground truth first
                acc_gt, correct_gt, total_gt = self.calculate_top1_csv_accuracy_vs_gt(csv_predictions, filter_by_llm_subset=llm_subset, annotations=annotations)
                accuracies["gt"] = acc_gt

                # Overall accuracy (A group)
                accuracies["A"] = self.calculate_top1_accuracy_vs_csv(annotations, csv_predictions, filter_by_llm_subset=llm_subset)
                
                # Only calculate accuracies for main groups (S, M), not subgroups
                group_names = ["S", "S+", "S-", "M", "M+", "M-"]
                for group_name in group_names:
                    if group_name in groups and groups[group_name]:  # Only calculate if group exists and is not empty
                        accuracies[group_name] = self.calculate_top1_accuracy_vs_csv(groups[group_name], csv_predictions, filter_by_llm_subset=llm_subset)
                    else:
                        accuracies[group_name] = 0.0
                
                # Calculate Im^Re: intersection of correctness sets (correct under both ImGT and ReGT)
                csv_like_preds_k = {img_id: [v] for img_id, v in csv_predictions.items()}
                im_re_acc, im_re_correct, im_re_total = self.calculate_im_intersect_re(annotations, csv_like_preds_k)
                accuracies["Im^Re"] = im_re_acc

                # Log detailed information
                self.logger.info(f"Image categorization complete (CSV predictions) for k={k}:")
                for group_name, size in group_sizes.items():
                    self.logger.info(f"  {group_name}: {size} images")

                results.append(AccuracyResults(
                    total_images=len(annotations),
                    group_sizes=group_sizes,
                    accuracies=accuracies,
                ))
            
            if multilabel_stats or flag_stats or biggest_bbox_stats:
                return None
            return results  # Return list of results for each k
        
        # Load CSV predictions
        csv_predictions = self.load_csv_predictions(csv_path)
        
        # If multilabel_stats mode is enabled, calculate and print stats then return
        if multilabel_stats:
            print("\n" + "="*80)
            print("MULTILABEL STATISTICS - Accuracy by unique label count")
            print("="*80)
            # Convert single predictions to list format
            csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
            self.calculate_multilabel_stats(annotations, csv_like_preds)
            return None
        
        # If flag_stats mode is enabled, calculate and print flag stats for S category then return
        if flag_stats:
            print("\n" + "="*80)
            print("FLAG STATISTICS - Accuracy for S category (single label) by bbox flags")
            print("="*80)
            s_annotations = groups.get("S", [])
            csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
            self.calculate_flag_stats(s_annotations, csv_like_preds)
            return None

        # If biggest_bbox_stats mode is enabled, calculate and print biggest bbox stats then return
        if biggest_bbox_stats:
            if biggest_bbox_stats == "adaptive":
                # For adaptive mode, load all top-k predictions
                csv_like_preds = self.load_csv_predictions_topk(csv_path, k=5)
            else:
                # For cumulative/single modes, only need top-1
                csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
            self.calculate_biggest_bbox_stats(annotations, csv_like_preds, image_base_path, mode=biggest_bbox_stats)
            return None

        # Calculate accuracies for each group
        accuracies = {}
        
        # Calculate accuracy vs ground truth first
        acc_gt, correct_gt, total_gt = self.calculate_top1_csv_accuracy_vs_gt(csv_predictions, filter_by_llm_subset=llm_subset, annotations=annotations)
        accuracies["gt"] = acc_gt

        # Overall accuracy (A group)
        accuracies["A"] = self.calculate_top1_accuracy_vs_csv(annotations, csv_predictions, filter_by_llm_subset=llm_subset)
        
        # Only calculate accuracies for main groups (S, M), not subgroups
        group_names = ["S", "S+", "S-", "M", "M+", "M-"]
        for group_name in group_names:
            if group_name in groups and groups[group_name]:  # Only calculate if group exists and is not empty
                accuracies[group_name] = self.calculate_top1_accuracy_vs_csv(groups[group_name], csv_predictions, filter_by_llm_subset=llm_subset)
            else:
                accuracies[group_name] = 0.0
        
        # Calculate Im^Re: intersection of correctness sets (correct under both ImGT and ReGT)
        csv_like_preds = {k: [v] for k, v in csv_predictions.items()}
        im_re_acc, im_re_correct, im_re_total = self.calculate_im_intersect_re(annotations, csv_like_preds)
        accuracies["Im^Re"] = im_re_acc

        return AccuracyResults(
            total_images=len(annotations),
            group_sizes=group_sizes,
            accuracies=accuracies,
        )

    def calculate_llm_acc_with_CI(self, file_names, class_lookup: Union[List | None], CI=0.95):
        acc_list = []
        for file_name in file_names:
            acc_list.append(self.calculate_llm_acc(file_name, class_lookup, folder_name="var"))
        
        acc_array = np.array(acc_list)
        mean_acc = np.mean(acc_array, axis=0)
        std_dev = np.std(acc_array, axis=0, ddof=1)
        n = len(file_names)

        assert n == 31, f"Expected 31 files, got {len(file_names)}"

        sem = std_dev / np.sqrt(n)
    
        # Critical value from the Student’s t-distribution
        t_crit = t.ppf(1 - (1 - CI) / 2, df=n - 1)

        margin_error = t_crit * sem

        print(f"{CI*100:.1f}% Confidence Interval for Mean Accuracy:")
        print(f"  - Mean accuracy: {mean_acc}")
        print(f"  - Error margin:   {margin_error}")
        
        label = r"{\scriptsize(Column Name)}"

        # Round values
        mean_rounded = np.round(mean_acc, 2)
        err_rounded = np.round(margin_error, 2)

        # Build formatted LaTeX string
        formatted_values = " & ".join([
            f"{m:.2f} \\textsuperscript{{\\scriptsize \\textcolor{{gray}}{{±{e:.2f}}}}}"
            for m, e in zip(mean_rounded, err_rounded)
        ])

        latex_line = f"{label} & {formatted_values} \\\\"

        print("\nLaTeX-formatted line:")
        print(latex_line)
        

    def calculate_llm_acc(self, file_name, class_lookup: Union[List | None], folder_name: str = "chatgpt", multilabel_stats: bool = False, flag_stats: bool = False, biggest_bbox_stats: str = None, image_base_path: str = None):
        annotations = self.load_annotations()
        self.class_lookup = class_lookup
        groups = self.categorize_images(annotations)
        group_sizes = {group: len(anns) for group, anns in groups.items()}

        with open(Path(project_root) / "data" / "raw" / folder_name / file_name, 'r', encoding='utf-8') as f:
            preds_json = json.load(f)
        
        self.llm_image_subset = {img_id for img_id in preds_json["predictions"].keys()}
        print(f"LLM predictions contain {len(self.llm_image_subset)} images")
        
        report_acc = []
        # Also save the mapped predictions JSON next to the original file
        mapped_out_path = (
            Path(project_root)
            / "data" / "raw" / folder_name
            / f"{Path(file_name).stem}_mapped_ids.json"
        ) if folder_name else None
        
        mapped_out_path = None
        
        csv_like_preds, unknown_counts, confidences = self._convert_llm_json_to_predictions(
            preds_json, save_path=mapped_out_path
        )
        unknown_before, unknown_after = unknown_counts
        
        # If multilabel_stats mode is enabled, calculate and print stats then return
        if multilabel_stats:
            print("\n" + "="*80)
            print("MULTILABEL STATISTICS - Accuracy by unique label count")
            print("="*80)
            self.calculate_multilabel_stats(annotations, csv_like_preds)
            return None
        
        # If flag_stats mode is enabled, calculate and print flag stats for S category then return
        if flag_stats:
            print("\n" + "="*80)
            print("FLAG STATISTICS - Accuracy for S category (single label) by bbox flags")
            print("="*80)
            s_annotations = groups.get("S", [])
            self.calculate_flag_stats(s_annotations, csv_like_preds)
            return None

        # If biggest_bbox_stats mode is enabled, calculate and print biggest bbox stats then return
        if biggest_bbox_stats:
            self.calculate_biggest_bbox_stats(annotations, csv_like_preds, image_base_path, mode=biggest_bbox_stats)
            return None

        acc_gt, correct_gt, total_gt = self.calculate_top1_llm_accuracy_vs_gt(csv_like_preds, annotations, evaluate_intersection_only=True)
        report_acc.append(acc_gt)
        if confidences:
            acc_ann, correct_ann, total_ann, avg_confidence = self.calculate_top1_accuracy_vs_csv(annotations, csv_like_preds, return_counts=True, confidences=confidences)
        else:
            acc_ann, correct_ann, total_ann = self.calculate_top1_accuracy_vs_csv(annotations, csv_like_preds, return_counts=True)
        report_acc.append(acc_ann)

        # Count unknowns only in the annotation image list
        ann_image_names = {a.image_name for a in annotations}
        unknown_after_in_annotations = sum(1 for img_id in ann_image_names if img_id in csv_like_preds and -1 in csv_like_preds[img_id])

        print("LLM accuracy summary:")
        print(f"  - Acc@1 vs ground truth: {acc_gt:.2f}% ({correct_gt}/{total_gt})")
        print(f"  - Acc@1 vs annotations:  {acc_ann:.2f}% ({correct_ann}/{total_ann})")
        
        if unknown_before > 0:
            print(f"  - Unknown class names before filtering: {unknown_before}")
        if unknown_after_in_annotations > 0:
            print(f"  - Unknown class names after filtering: {unknown_after_in_annotations}")

        group_names = ["S", "S+", "S-", "M", "M+", "M-", "N"]
        for group_name in group_names:
            anns = groups.get(group_name, [])
            if anns:
                if confidences:
                    g_acc, g_correct, g_total, g_avg_conf = self.calculate_top1_accuracy_vs_csv(anns, csv_like_preds, evaluate_intersection_only=True, export_file=group_name, return_counts=True, confidences=confidences)
                else:
                    g_acc, g_correct, g_total = self.calculate_top1_accuracy_vs_csv(anns, csv_like_preds, evaluate_intersection_only=True, export_file=group_name, return_counts=True)
                report_acc.append(g_acc) if group_name != "N" else None
                
                # Count unknown class names in this group
                unknown_count = self._count_unknown_in_group(anns, csv_like_preds)
                print(f"    - {group_name}: {g_acc:.2f}% ({g_correct}/{g_total}), unknown class names: {unknown_count}")
                
                # Print correctly classified images for S- and M- groups
                if group_name in ["S-", "M-"] and g_correct > 0:
                    correct_images = self._get_correctly_classified_images(anns, csv_like_preds)
                    if correct_images:
                        ...
                        #print(f"      Correctly classified {group_name} images: {', '.join(correct_images)}")
                
                gt_g_acc, _, _ = self.calculate_top1_llm_accuracy_vs_gt(csv_like_preds, anns, evaluate_intersection_only=True)
                #print(f"      - vs GT: {gt_g_acc:.2f}%")
            else:
                print(f"    - {group_name}: N/A (0 images), unknown class names: 0")

        #assert len(report_acc) == 8, f"Expected 8 accuracy values, got {len(report_acc)}"
        report_acc.append(unknown_after_in_annotations)

        # Calculate Im^Re: intersection of correctness sets (correct under both ImGT and ReGT)
        im_re_acc, im_re_correct, im_re_total = self.calculate_im_intersect_re(annotations, csv_like_preds)
        print(f"    - Im^Re: {im_re_acc:.2f}% ({im_re_correct}/{im_re_total})")
        report_acc.append(im_re_acc)

        print(" & ".join(
            str(int(f)) if isinstance(f, int) else f"{f:.2f}"
            for f in report_acc
        ))
        return report_acc

    def calculate_im_intersect_re(self, annotations: List[ImageAnnotation],
                                  csv_predictions: Dict[str, List[int]]) -> Tuple[float, int, int]:
        """
        Calculate Im^Re: accuracy on the intersection of the two correctness sets.

        Returns the percentage of images that are correct under BOTH the original
        ImageNet ground truth (ImGT) and the reannotated ground truth (ReGT).

        Args:
            annotations: List of image annotations
            csv_predictions: Dict[img_id -> predicted class id(s)]

        Returns:
            Tuple of (accuracy_percent, num_correct_both, num_evaluated)
        """
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())

        correct_both = 0
        total = 0

        for img_id in common_ids:
            ann = ann_by_name[img_id]
            predicted_ids = csv_predictions[img_id]
            gt_label = imagenet_classes.val_image_to_1k_label(img_id)
            if gt_label not in range(1000):
                continue
            total += 1

            # Check correctness under ImGT
            correct_im = False
            for pred_id in predicted_ids:
                if self.is_correct(gt_label, int(pred_id)):
                    correct_im = True
                    break

            # Check correctness under ReGT
            correct_re = False
            for pred_id in predicted_ids:
                membership_match = self.is_correct(ann.reannotated_labels, pred_id)
                if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                    correct_re = True
                    break

            if correct_im and correct_re:
                correct_both += 1

        acc = (correct_both / total * 100) if total > 0 else 0.0
        return round(acc, 2), correct_both, total

    def _count_unknown_in_group(self, annotations: List[ImageAnnotation],
                               csv_predictions: Dict[str, List[int]]) -> int:
        """
        Count the number of images in a group that have unknown class names (prediction = -1).
        
        Args:
            annotations: List of image annotations for the group
            csv_predictions: Dict[img_id -> predicted class id(s)]
            
        Returns:
            Number of images with unknown class names
        """
        unknown_count = 0
        ann_by_name = {a.image_name: a for a in annotations}
        
        for img_id in ann_by_name.keys():
            if img_id in csv_predictions:
                predicted_ids = csv_predictions[img_id]
                # Check if any prediction is -1 (unknown)
                if -1 in predicted_ids:
                    unknown_count += 1
                    
        return unknown_count

    def _get_correctly_classified_images(self, annotations: List[ImageAnnotation], 
                                          csv_predictions: Dict[str, List[int]]) -> List[str]:
        """
        Get list of image names that were correctly classified.
        
        Args:
            annotations: List of image annotations
            csv_predictions: Dict[img_id -> predicted class id(s)]
            
        Returns:
            List of correctly classified image names
        """
        correct_images = []
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())
        
        for img_id in common_ids:
            ann = ann_by_name[img_id]
            predicted_ids = csv_predictions[img_id]
            
            for predicted_class in predicted_ids:
                membership_match = self.is_correct(ann.reannotated_labels, predicted_class)
                # Correct or OOD image or Not Sure image
                if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                    correct_images.append(img_id)
                    break
        
        return sorted(correct_images)

    def calculate_multilabel_stats(self, annotations: List[ImageAnnotation], 
                                   csv_predictions: Dict[str, List[int]]) -> None:
        """
        Calculate and print accuracies separately for images grouped by unique label count
        and for images with crowd boxes.
        
        For each image, counts the number of unique labels (ignoring -1 "not sure" labels).
        If an image has 4 bounding boxes but they all have the same label, it counts as 1 label.
        
        Args:
            annotations: List of image annotations
            csv_predictions: Dict[img_id -> predicted class id(s)]
        """
        from collections import defaultdict
        
        # Group annotations by unique label count
        # Key: number of unique labels, Value: list of annotations
        label_count_groups: Dict[int, List[ImageAnnotation]] = defaultdict(list)
        crowd_annotations: List[ImageAnnotation] = []
        n_images_count = 0  # Count of images with 0 valid labels (N group)
        
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())
        
        for img_id in common_ids:
            ann = ann_by_name[img_id]
            
            # Check if any bbox has crowd flag
            has_crowd = any(bbox.crowd_flag for bbox in ann.bboxes)
            if has_crowd:
                crowd_annotations.append(ann)
            
            # Count unique semantic labels (excluding -1, collapsing equivalent classes)
            unique_labels = {label for label in ann.reannotated_labels if label != -1}
            if self.class_lookup is not None:
                seen: Set[int] = set()
                num_unique = 0
                for lbl in unique_labels:
                    if lbl not in seen:
                        num_unique += 1
                        seen.add(lbl)
                        seen.update(self.class_lookup.get_neighbors(lbl))
            else:
                num_unique = len(unique_labels)
            
            # Group by unique label count
            if num_unique > 0:
                label_count_groups[num_unique].append(ann)
            else:
                n_images_count += 1
        
        # Print count of N images (0 valid labels)
        print(f"N images (0 labels): {n_images_count}")
        
        # Calculate and print accuracies for each label count
        max_labels = max(label_count_groups.keys()) if label_count_groups else 0
        latex_accs = []

        for num_labels in range(1, max_labels + 1):
            anns = label_count_groups.get(num_labels, [])
            if anns:
                acc, correct, total = self._calculate_accuracy_for_subset(anns, csv_predictions)
                print(f"{num_labels} label{'s' if num_labels > 1 else ''}: {acc:.2f}% ({correct}/{total})")
                latex_accs.append(f"{acc:.2f}")
            else:
                print(f"{num_labels} label{'s' if num_labels > 1 else ''}: N/A (0/0)")
                latex_accs.append("N/A")

        # Calculate and print accuracy for crowd images
        if crowd_annotations:
            acc, correct, total = self._calculate_accuracy_for_subset(crowd_annotations, csv_predictions)
            print(f"crowd: {acc:.2f}% ({correct}/{total})")
            latex_accs.append(f"{acc:.2f}")
        else:
            print(f"crowd: N/A (0/0)")
            latex_accs.append("N/A")

        print(" & ".join(latex_accs))

        # Second latex line: 1-5 individual + weighted average for >=6
        latex_accs_compact = []
        for num_labels in range(1, 6):
            anns = label_count_groups.get(num_labels, [])
            if anns:
                acc, correct, total = self._calculate_accuracy_for_subset(anns, csv_predictions)
                latex_accs_compact.append(f"{acc:.2f}")
            else:
                latex_accs_compact.append("N/A")

        anns_6plus = [ann for n, anns in label_count_groups.items() if n >= 6 for ann in anns]
        if anns_6plus:
            acc, correct, total = self._calculate_accuracy_for_subset(anns_6plus, csv_predictions)
            latex_accs_compact.append(f"{acc:.2f}")
        else:
            latex_accs_compact.append("N/A")

        print(" & ".join(latex_accs_compact))
    
    def _calculate_accuracy_for_subset(self, annotations: List[ImageAnnotation], 
                                       csv_predictions: Dict[str, List[int]]) -> Tuple[float, int, int]:
        """
        Calculate Top-1 accuracy for a subset of annotations.
        
        Args:
            annotations: List of image annotations
            csv_predictions: Dict[img_id -> predicted class id(s)]
            
        Returns:
            Tuple of (accuracy_percent, num_correct, num_total)
        """
        correct = 0
        total = 0
        
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())
        
        for img_id in common_ids:
            ann = ann_by_name[img_id]
            predicted_ids = csv_predictions[img_id]
            total += 1
            
            for predicted_class in predicted_ids:
                membership_match = self.is_correct(ann.reannotated_labels, predicted_class)
                # Correct if prediction matches any reannotated label, or if OOD/Not Sure
                if membership_match or len(ann.reannotated_labels) == 0 or (len(ann.reannotated_labels) == 1 and list(ann.reannotated_labels)[0] == -1):
                    correct += 1
                    break
        
        acc = (correct / total * 100) if total > 0 else 0.0
        return round(acc, 2), correct, total

    def calculate_biggest_bbox_stats(self, annotations: List[ImageAnnotation],
                                     csv_predictions: Dict[str, List[int]],
                                     image_base_path: str = None,
                                     mode: str = "cumulative",
                                     max_k: int = 10) -> None:
        """
        Calculate accuracy considering only the k biggest bboxes as valid ground truth.

        Three modes are supported:
        - "cumulative": k largest bboxes as valid GT (k=1 means biggest only, k=2 means 2 biggest, etc.)
        - "single": only the k-th largest bbox as valid GT (k=1 means biggest, k=2 means 2nd biggest, etc.)
        - "adaptive": k-th largest bbox label checked against model's top-m predictions,
                      where m = min(num_model_predictions, num_regt_labels)

        For multilabel images (>1 unique label), we sort bboxes by area and check if
        the model's prediction(s) match the valid labels.

        Reports accuracy for k=1,2,...,10 where:
        - Cumulative mode: k=1 only biggest, k=2 two biggest, etc. (model top-1 vs k labels)
        - Single mode: k=1 only biggest, k=2 only 2nd biggest, etc. (model top-1 vs 1 label)
        - Adaptive mode: k=1 only biggest, k=2 only 2nd biggest, etc. (model top-m vs 1 label)

        This helps understand if models identify the main/prominent objects vs small background objects.

        Args:
            annotations: List of image annotations
            csv_predictions: Dict[img_id -> predicted class id(s)]
            image_base_path: Base path to ImageNet validation images (e.g., '/path/to/val')
                            If None, will try to use ANNOTATIONS_ROOT_FOLDER from settings
            mode: "cumulative" (k largest bboxes), "single" (only k-th largest bbox),
                  or "adaptive" (k-th bbox vs model's top-m predictions)
        """
        from config.settings import ANNOTATIONS_ROOT_FOLDER

        if image_base_path is None:
            image_base_path = ANNOTATIONS_ROOT_FOLDER

        if mode == "cumulative":
            mode_desc = "k largest bboxes"
        elif mode == "single":
            mode_desc = "k-th largest bbox only"
        else:  # adaptive
            mode_desc = "k-th largest bbox vs model's top-m preds (m=min(num_preds, num_labels))"
        print(f"\nBIGGEST BBOX ACCURACY STATISTICS (Mode: {mode})")
        print("=" * 60)
        print(f"Accuracy when considering {mode_desc} as valid GT")
        print("(Only for multilabel images with >1 unique label)")
        print("=" * 60)

        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())

        # Filter to multilabel images only
        multilabel_annotations = []
        for img_id in common_ids:
            ann = ann_by_name[img_id]
            unique_labels = {label for label in ann.reannotated_labels if label != -1}
            if len(unique_labels) > 1:
                multilabel_annotations.append(ann)

        print(f"\nTotal multilabel images: {len(multilabel_annotations)}")

        # Track results for each k
        results = {k: {"correct": 0, "total": 0} for k in range(1, max_k + 1)}
        dominant_result = {"correct": 0, "total": 0}
        single_dominant_result = {"correct": 0, "total": 0}
        images_with_area_computed = 0
        images_skipped = 0

        for ann in tqdm(multilabel_annotations, desc="Computing bbox areas", unit="images"):
            img_id = ann.image_name
            predicted_ids = csv_predictions.get(img_id, [])

            if not predicted_ids:
                continue

            # Get model's top-1 prediction (for cumulative/single modes)
            top1_pred = predicted_ids[0]

            # For adaptive mode, calculate m = min(num_model_preds, num_regt_labels)
            num_model_preds = len(predicted_ids)
            num_regt_labels = len({label for label in ann.reannotated_labels if label != -1})
            m_adaptive = min(num_model_preds, num_regt_labels)
            top_m_preds = set(predicted_ids[:m_adaptive])  # Model's top-m predictions

            # Dominant-object subsets
            if mode == "single":
                dominant_labels = {
                    label
                    for bbox in ann.bboxes
                    if bbox.dominant_object
                    for label in bbox.labels
                    if label != -1
                }
                if dominant_labels:
                    dominant_result["total"] += 1
                    if self.is_correct(dominant_labels, top1_pred):
                        dominant_result["correct"] += 1
                    # Single-dominant-label subset: all dominant bboxes share exactly one unique label
                    if len(dominant_labels) == 1:
                        single_dominant_result["total"] += 1
                        if self.is_correct(dominant_labels, top1_pred):
                            single_dominant_result["correct"] += 1

            # Calculate bbox areas and sort by area (descending)
            bbox_areas = []
            try:
                # Build image path
                img_class_folder = imagenet_classes.val_image_to_21k_key(img_id)
                img_path = os.path.join(image_base_path, img_class_folder, img_id)

                if not os.path.exists(img_path):
                    # Try without class folder (flat structure)
                    img_path = os.path.join(image_base_path, img_id)

                if not os.path.exists(img_path):
                    images_skipped += 1
                    continue

                img = Image.open(img_path)
                img_width, img_height = img.width, img.height
                img_area = img_width * img_height

                for bbox in ann.bboxes:
                    # Skip "not sure" labels (check if all labels in the box are -1)
                    if bbox.labels == {-1}:
                        continue
                    # Filter out -1 from the labels set
                    valid_labels = {l for l in bbox.labels if l != -1}
                    if not valid_labels:
                        continue

                    coords = bbox.coordinates
                    if len(coords) != 4:
                        continue

                    x_min, y_min, x_max, y_max = coords
                    # Clip to image bounds
                    x_min = max(0, min(x_min, img_width))
                    y_min = max(0, min(y_min, img_height))
                    x_max = max(0, min(x_max, img_width))
                    y_max = max(0, min(y_max, img_height))

                    area = (x_max - x_min) * (y_max - y_min)
                    if area > 0:
                        # Store area and ALL labels for this box (handles grouped boxes)
                        bbox_areas.append((area, valid_labels))

                images_with_area_computed += 1

            except Exception as e:
                self.logger.warning(f"Could not compute bbox areas for {img_id}: {e}")
                images_skipped += 1
                continue

            if not bbox_areas:
                continue

            # Sort by area descending (biggest first)
            bbox_areas.sort(key=lambda x: x[0], reverse=True)

            # For each k, check if prediction is correct based on mode
            for k in range(1, max_k + 1):
                if mode == "cumulative":
                    # Cumulative: Get unique labels from the k largest bboxes
                    k_valid_labels = set()
                    for i, (area, box_labels) in enumerate(bbox_areas):
                        if i >= k:
                            break
                        k_valid_labels.update(box_labels)  # Union all labels from this box

                    if not k_valid_labels:
                        continue

                    results[k]["total"] += 1
                    # Check if top-1 prediction matches any of the k largest bbox labels
                    if self.is_correct(k_valid_labels, top1_pred):
                        results[k]["correct"] += 1

                elif mode == "single":
                    # Single: Get labels from only the k-th largest bbox
                    if k <= len(bbox_areas):
                        k_valid_labels = bbox_areas[k-1][1]  # k-1 because 0-indexed, already a set
                    else:
                        continue  # Not enough bboxes

                    results[k]["total"] += 1
                    # Check if top-1 prediction matches any label of the k-th bbox
                    if self.is_correct(k_valid_labels, top1_pred):
                        results[k]["correct"] += 1

                else:  # adaptive mode
                    # Adaptive: Check if any label of k-th largest bbox is in model's top-m predictions
                    if k <= len(bbox_areas):
                        kth_bbox_labels = bbox_areas[k-1][1]  # k-1 because 0-indexed, now a set
                    else:
                        continue  # Not enough bboxes

                    results[k]["total"] += 1
                    # Check if any of the model's top-m predictions matches any label of the k-th bbox
                    is_match = False
                    for pred in top_m_preds:
                        if self.is_correct(kth_bbox_labels, pred):
                            is_match = True
                            break
                    if is_match:
                        results[k]["correct"] += 1

        # Print results
        print(f"\nImages processed: {images_with_area_computed}")
        print(f"Images skipped (no image file): {images_skipped}")

        if mode == "single":
            sdom_total = single_dominant_result["total"]
            sdom_correct = single_dominant_result["correct"]
            sdom_acc = (sdom_correct / sdom_total * 100) if sdom_total > 0 else 0.0
            print(f"\nSingle-dominant-label subset (dominant bboxes all share exactly one label):")
            print(f"  Images in subset: {sdom_total}")
            print(f"  Accuracy (pred matches the single dominant label): {sdom_acc:.2f}%  ({sdom_correct}/{sdom_total})")
            self.logger.info(f"Single-dominant-label subset: acc={sdom_acc:.2f}% ({sdom_correct}/{sdom_total})")

            dom_total = dominant_result["total"]
            dom_correct = dominant_result["correct"]
            dom_acc = (dom_correct / dom_total * 100) if dom_total > 0 else 0.0
            print(f"\nDominant-object subset (multilabel images with >=1 dominant bbox):")
            print(f"  Images in subset: {dom_total}")
            print(f"  Accuracy (pred matches any dominant bbox label): {dom_acc:.2f}%  ({dom_correct}/{dom_total})")
            self.logger.info(f"Dominant-object subset: acc={dom_acc:.2f}% ({dom_correct}/{dom_total})")

        if mode == "cumulative":
            print("\nResults (k = number of largest bboxes considered as valid GT):")
        elif mode == "single":
            print("\nResults (k = rank of single bbox considered as valid GT, 1=largest):")
        else:  # adaptive
            print("\nResults (k = rank of bbox, checked against model's top-m preds, m=min(num_preds, num_labels)):")
        print("-" * 40)
        print(f"{'k':<5} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-" * 40)

        for k in range(1, max_k + 1):
            total = results[k]["total"]
            correct = results[k]["correct"]
            acc = (correct / total * 100) if total > 0 else 0.0
            print(f"{k:<5} {acc:>6.2f}%      {correct}/{total}")

        # Also print as a single line for easy copy-paste to table
        print(f"\nFor table (k=1 to k={max_k}):")
        acc_values = []
        if mode == "single":
            acc_values.append(f"{sdom_acc:.2f}")
            acc_values.append(f"{dom_acc:.2f}")
        for k in range(1, max_k + 1):
            total = results[k]["total"]
            correct = results[k]["correct"]
            acc = (correct / total * 100) if total > 0 else 0.0
            acc_values.append(f"{acc:.2f}")
        print(" & ".join(acc_values))

        ret = {"k": {k: results[k] for k in range(1, max_k + 1)}}
        if mode == "single":
            ret["dom_acc"]  = dom_acc
            ret["sdom_acc"] = sdom_acc
        return ret

    def calculate_flag_stats(self, annotations: List[ImageAnnotation],
                             csv_predictions: Dict[str, List[int]]) -> None:
        """
        Calculate and print accuracies for S category (single label) images grouped by bbox flags.
        
        Groups images by the flags present on their bboxes:
        - no_flags: images where no bbox has any flag
        - crowd: images where bboxes only have crowd flag (no other flags)
        - ocr_needed: images where bboxes only have ocr_needed flag
        - rendition: images where bboxes only have rendition flag
        - reflected: images where bboxes only have reflected flag
        - mixed: images where bboxes have multiple different flags, or a single bbox has multiple flags
        
        Args:
            annotations: List of S category image annotations (single label images)
            csv_predictions: Dict[img_id -> predicted class id(s)]
        """
        # Groups for each flag type
        flag_groups = {
            "no_flags": [],
            "crowd": [],
            "ocr_needed": [],
            "rendition": [],
            "reflected": [],
            "mixed": [],
        }
        
        ann_by_name = {a.image_name: a for a in annotations}
        common_ids = set(ann_by_name.keys()) & set(csv_predictions.keys())
        
        for img_id in common_ids:
            ann = ann_by_name[img_id]
            
            # Collect all flags from all bboxes in this image
            # For each bbox, determine which flags are set
            image_flags = set()  # Set of flag names present in the image
            has_mixed_bbox = False  # True if any single bbox has multiple flags
            
            for bbox in ann.bboxes:
                bbox_flags = []
                if bbox.crowd_flag:
                    bbox_flags.append("crowd")
                if bbox.ocr_needed_flag:
                    bbox_flags.append("ocr_needed")
                if bbox.rendition_flag:
                    bbox_flags.append("rendition")
                if bbox.reflected_flag:
                    bbox_flags.append("reflected")
                
                # Check if this single bbox has multiple flags
                if len(bbox_flags) > 1:
                    has_mixed_bbox = True
                
                image_flags.update(bbox_flags)
            
            # Categorize the image based on its flags
            if has_mixed_bbox or len(image_flags) > 1:
                # Mixed: either a bbox has multiple flags, or different bboxes have different flags
                flag_groups["mixed"].append(ann)
            elif len(image_flags) == 0:
                # No flags on any bbox
                flag_groups["no_flags"].append(ann)
            else:
                # Exactly one type of flag across all bboxes
                flag_type = list(image_flags)[0]
                flag_groups[flag_type].append(ann)
        
        # Print accuracies for each flag group
        flag_order = ["no_flags", "crowd", "ocr_needed", "rendition", "reflected", "mixed"]
        flag_labels = {
            "no_flags": "No flags",
            "crowd": "Crowd",
            "ocr_needed": "OCR needed",
            "rendition": "Rendition",
            "reflected": "Reflected",
            "mixed": "Mixed flags",
        }
        
        for flag_type in flag_order:
            anns = flag_groups[flag_type]
            if anns:
                acc, correct, total = self._calculate_accuracy_for_subset(anns, csv_predictions)
                print(f"{flag_labels[flag_type]}: {acc:.2f}% ({correct}/{total})")
            else:
                print(f"{flag_labels[flag_type]}: N/A (0/0)")

    def calculate_top1_llm_accuracy_vs_gt(self, predictions: Dict[str, List[int]], annotations=None, evaluate_intersection_only=False) -> Tuple[float, int, int]:
        """Compute Top-1 accuracy vs ImageNet ground-truth labels for LLM predictions.

        Args:
            predictions: Mapping of image_id -> class_id (use -1 for unknown classes)

        Returns:
            Tuple of (accuracy_percent, num_correct, num_evaluated)
        """
        if evaluate_intersection_only:
            # Strict intersection of keys
            ann_by_name = {a.image_name: a for a in annotations}
            common_ids = set(ann_by_name.keys()) & set(predictions.keys())

        num_correct = 0
        total = 0
        for img_id, pred_ids in predictions.items():
            if evaluate_intersection_only and img_id not in common_ids:
                continue
            gt_label = imagenet_classes.val_image_to_1k_label(img_id)
            if gt_label not in range(1000):
                continue
            total += 1
            for pred_id in pred_ids:
                if self.is_correct(gt_label, int(pred_id)):
                    num_correct += 1
                    break

        acc = (num_correct / total * 100) if total > 0 else 0.0
        print(f"{num_correct} correct out of {total} evaluated against GT")
        return round(acc, 2), num_correct, total

    def calculate_top1_csv_accuracy_vs_gt(self, predictions: Dict[str, int], filter_by_llm_subset: bool = False,
                                          annotations: List[ImageAnnotation] = None) -> \
    Tuple[float, int, int]:
        """Compute Top-1 accuracy vs ImageNet ground-truth labels for CSV predictions.

        Args:
            predictions: Mapping of image_id -> class_id (use -1 for unknown classes)
            filter_by_llm_subset: If True, only evaluate images in self.llm_image_subset
            annotations: If provided, restrict evaluation to the intersection of
                        prediction image IDs and annotation image IDs so that ImGT
                        accuracy is computed on the same subset as ReGT accuracy.

        Returns:
            Tuple of (accuracy_percent, num_correct, num_evaluated)
        """
        # Filter predictions by LLM subset if requested
        if filter_by_llm_subset and self.llm_image_subset:
            predictions = {k: v for k, v in predictions.items() if k in self.llm_image_subset}

        # Filter predictions to annotation subset so ImGT is evaluated on the same
        # images as ReGT (instead of all 50k val images)
        if annotations is not None:
            ann_image_ids = {a.image_name for a in annotations}
            predictions = {k: v for k, v in predictions.items() if k in ann_image_ids}

        num_correct = 0
        total = 0
        for img_id, pred_id in predictions.items():
            gt_label = imagenet_classes.val_image_to_1k_label(img_id)
            if gt_label not in range(1000):
                continue
            total += 1
            if self.is_correct(gt_label, int(pred_id)):
                num_correct += 1

        acc = (num_correct / total * 100) if total > 0 else 0.0
        print(f"{num_correct} correct out of {total} evaluated against GT")
        return round(acc, 2), num_correct, total

    def _convert_llm_json_to_predictions(
        self,
        predictions_json: Dict,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, List[int]], Tuple[int, int], Dict[str, float]]:
        """Convert LLM predictions JSON to a dict of image_id -> [class_ids].

        Supports:
        - String/int class id: "187" or 187 -> [187]
        - Comma-separated class ids: "639,445" -> [639, 445]
        - Semicolon-separated class names: "dog;cat" -> mapped to ids
        - String class name: "dog" -> mapped to [class_id] or [-1] if unknown
        - List of ids or names: ["dog", "cat"] or [639, 445] -> list of ids
        - Empty string/list/None -> [-1] (added to unknown list as "")
        - Alt format (no mapping key):
              {
                "ILSVRC2012_val_00046166.JPEG": {"label": 820, "similarity": 1.0},
                ...
              }
          or nested under "predictions" key. In this case, each value dict must
          contain a "label" field (int/str or list) which is used directly (no
          name -> id mapping necessary).

        Args:
            predictions_json: Input predictions JSON (possibly with free-text labels).
            save_path: If provided, saves a mapped JSON next to the original, with a
                top-level "predictions" key mapping each image to a single class id
                (first mapped id if multiple) or -1 when unknown.

        Returns:
            predictions_dict: Dict[image_id, List[class_ids]]
            unknown_counts: Tuple[int, int] = (unknown_before_filtering, unknown_after_filtering)
            confidences: Dict[image_id, float] with probability/confidence if provided
        """
        # Use a helper to allow manual overrides for specific classes (356-359)
        cls_names = [imagenet_classes.get_1k_clean_name(i).lower() for i in range(1000)]
        cls_name2label = {name: i for i, name in enumerate(cls_names)}




        preds: Dict[str, List[int]] = {}
        probs: Dict[str, float] = {}
        unknown: List[str] = []


        # Determine the raw predictions mapping. Preferred is predictions_json["predictions"].
        raw_preds = predictions_json.get("predictions")

        # If no explicit "predictions" key, attempt to interpret the top-level dict
        # as the predictions mapping (alternative format). We require that values are
        # dicts containing a 'label' key OR primitive / list values signifying labels.
        if raw_preds is None:
            if isinstance(predictions_json, dict):
                sample_values = list(predictions_json.values())
                if sample_values and all(
                    isinstance(v, (dict, list, str, int)) for v in sample_values
                ):
                    # Heuristic: treat as raw predictions
                    raw_preds = predictions_json
            if raw_preds is None:
                raw_preds = {}

        for pred_idx, (img_id, pred_value) in enumerate(raw_preds.items()):
            img_preds: List[int] = []

            if not pred_value:  # empty string, None, []
                preds[img_id] = [-1]
                unknown.append("")  # record empty string explicitly
                continue

            # If alt format: value is a dict with a 'label' field (and possibly other metadata)
            if isinstance(pred_value, dict) and "label" in pred_value:
                # Capture probability/confidence if available
                if "prob" in pred_value:
                    try:
                        probs[img_id] = float(pred_value["prob"])
                    except Exception:
                        pass
                '''elif "similarity" in pred_value:  # optional fallback
                    try:
                        probs[img_id] = float(pred_value["similarity"])
                    except Exception:
                        pass'''
                pred_value = pred_value["label"]

            if isinstance(pred_value, (int, str)):
                pred_value = str(pred_value).strip()

                # If it looks like numeric ids, use comma separator
                if any(ch.isdigit() for ch in pred_value) and "," in pred_value:
                    pred_value = [p.strip() for p in pred_value.split(",") if p.strip()]
                # Otherwise, if it looks like names, use semicolon separator
                elif ";" in pred_value:
                    pred_value = [p.strip() for p in pred_value.split(";") if p.strip()]
                else:
                    pred_value = [pred_value] if pred_value else []

            if isinstance(pred_value, list):
                for candidate in pred_value:
                    if isinstance(candidate, int) or (isinstance(candidate, str) and candidate.isdigit()):
                        # Pure class id
                        try:
                            img_preds.append(int(candidate))
                        except ValueError:
                            img_preds.append(-1)
                            unknown.append(str(candidate))
                    elif isinstance(candidate, str):
                        name_norm = candidate.lower().strip()
                        class_id = cls_name2label.get(name_norm, -1)
                        if class_id == -1:
                                unknown.append(name_norm)
                        img_preds.append(class_id)
            else:
                # Final fallback: single primitive (e.g., dict without 'label' after extraction, or unexpected type)
                if isinstance(pred_value, int):
                    img_preds.append(pred_value)
                elif isinstance(pred_value, str) and pred_value.isdigit():
                    img_preds.append(int(pred_value))
                elif isinstance(pred_value, str):
                    name_norm = pred_value.lower().strip()
                    class_id = cls_name2label.get(name_norm, -1)
                    if class_id == -1:
                            unknown.append(name_norm)
                    img_preds.append(class_id)

            if not img_preds:  # if list ends up empty
                img_preds = [-1]
                unknown.append("")  # treat empty as unknown

            preds[img_id] = img_preds

        # Count unknowns before filtering
        unknown_before_filter = len(unknown)
        
        unknown_after_filter = sum(1 for pred_list in preds.values() if -1 in pred_list)


        # Optionally save a JSON with the mapped predictions in original-style layout
        # Structure: { "predictions": { <img_id>: <single_int_id_or_-1>, ... } }
        if save_path is not None:
            try:
                to_save = {
                    "predictions": {
                        img_id: vals for img_id, vals in preds.items()
                    }
                }
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(to_save, f, indent=2)
                self.logger.info(f"Saved mapped predictions JSON to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save mapped predictions JSON to {save_path}: {e}")


        return preds, (unknown_before_filter, unknown_after_filter), probs

    def get_detailed_statistics_vs_gt(self) -> Dict:
        """
        Get detailed statistics about the annotations and accuracies against ground truth.
        
        Returns:
            Dictionary with detailed statistics
        """
        annotations = self.load_annotations()
        groups = self.categorize_images(annotations)
        
        stats = {
            "total_images": len(annotations),
            "groups": {}
        }
        
        for group_name, group_annotations in groups.items():
            group_stats = {
                "count": len(group_annotations),
                "percentage": len(group_annotations) / len(annotations) * 100 if annotations else 0,
                "accuracy": self.calculate_top1_accuracy_vs_gt(group_annotations),
                "sample_images": [ann.image_name for ann in group_annotations[:5]]  # First 5 as samples
            }
            stats["groups"][group_name] = group_stats
        
        # Overall accuracy
        stats["overall_accuracy"] = self.calculate_top1_accuracy_vs_gt(annotations)
        
        return stats
    
    def get_detailed_statistics_vs_csv(self, csv_path: Path) -> Dict:
        """
        Get detailed statistics about the annotations and accuracies against CSV predictions.
        
        Args:
            csv_path: Path to the CSV file with predictions
        
        Returns:
            Dictionary with detailed statistics
        """
        annotations = self.load_annotations()
        csv_predictions = self.load_csv_predictions(csv_path)
        groups = self.categorize_images(annotations)
        
        stats = {
            "total_images": len(annotations),
            "groups": {}
        }
        
        for group_name, group_annotations in groups.items():
            group_stats = {
                "count": len(group_annotations),
                "percentage": len(group_annotations) / len(annotations) * 100 if annotations else 0,
                "accuracy": self.calculate_top1_accuracy_vs_csv(group_annotations, csv_predictions),
                "sample_images": [ann.image_name for ann in group_annotations[:5]]  # First 5 as samples
            }
            stats["groups"][group_name] = group_stats
        
        # Overall accuracy
        stats["overall_accuracy"] = self.calculate_top1_accuracy_vs_csv(annotations, csv_predictions)
        
        return stats
    
    def is_correct(self, original_class: Union[int, List[int]], predicted_class: int) -> bool:
        if predicted_class == -1:
            return False  # Unknown prediction is always incorrect
        
        if isinstance(original_class, int):
            original_classes = [original_class]
        else:
            original_classes = original_class

        if self.class_lookup is not None:
            return self.class_lookup.is_in_equal_group_with_any(
                original_classes, predicted_class
            )
        else:
            print("!!! - - - No class lookup provided, using direct comparison.")
        return predicted_class in original_classes
    

