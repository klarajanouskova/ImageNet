"""
Class lookup builder for reannotation experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Union, Any
from collections import defaultdict


class ClassLookupBuilder:
    """
    Builder for fast class lookup structures from class update configurations.
    
    Creates efficient mappings for:
    - Equal classes (eq_classes): maps each class to all its neighbors in the same group
    """
    
    def __init__(self):
        """Initialize the class lookup builder."""
        self.logger = logging.getLogger(__name__)
        self.equal_class_lookup: Dict[int, Set[int]] = {}
        self.hier_classes: Dict[str, List[int]] = {}
        self.image_overrides: Dict[str, str] = {}  # image_name -> annotator_name
        
    def build_lookup_from_json(self, class_update_json: str) -> Dict[str, Any]:
        """
        Build fast lookup structure from class update JSON.
        
        Args:
            class_update_json: Path to JSON file, JSON string, or dictionary
            
        Returns:
            Dictionary containing the built lookup structures
        """
        self.logger.info("Building class lookup structure from JSON")
        
        # Parse JSON input
        if isinstance(class_update_json, dict):
            config = class_update_json
        elif isinstance(class_update_json, Path):
            if not Path(class_update_json).exists():
                raise ValueError("class_update_json must be a file path")
            with open(class_update_json, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"An incorrect filepath is specified: got {type(class_update_json)}")
        
        # Extract configuration sections
        eq_classes = config.get("eq_classes", {})
        hier_classes = config.get("hier_classes", {})
        image_overrides = config.get("image_overrides", {})
        
        # Build equal class lookup
        self._build_equal_class_lookup(eq_classes)
        
        # Preserve hierarchical classes as-is
        self.hier_classes = hier_classes.copy()
        
        # Store image overrides
        self.image_overrides = image_overrides.copy()
        
        # Return complete lookup structure
        return {
            "equal_class_lookup": self.equal_class_lookup,
            "hier_classes": self.hier_classes,
            "image_overrides": self.image_overrides,
            "metadata": config.get("metadata", {})
        }
    
    def _build_equal_class_lookup(self, eq_classes: Dict[str, List[int]]) -> None:
        """
        Build lookup structure for equal classes.
        
        Args:
            eq_classes: Dictionary mapping group names to lists of class IDs
        """
        self.logger.info(f"Building equal class lookup for {len(eq_classes)} groups")
        
        for group_name, class_list in eq_classes.items():
            if not class_list:
                continue
                
            # Convert to set for O(1) lookup
            class_set = set(class_list)
            
            # Map each class to all other classes in the same group
            for class_id in class_set:
                # Each class maps to all other classes in its group (excluding itself)
                neighbors = class_set - {class_id}
                self.equal_class_lookup[class_id] = neighbors
                
                self.logger.debug(f"Class {class_id} in group '{group_name}' maps to neighbors: {neighbors}")
        
        self.logger.info(f"Built equal class lookup for {len(self.equal_class_lookup)} classes")
    
    def get_neighbors(self, class_id: int) -> Set[int]:
        """
        Get all neighbors for a given class ID.
        
        Args:
            class_id: The class ID to find neighbors for
            
        Returns:
            Set of neighbor class IDs
        """
        return self.equal_class_lookup.get(class_id, set())
    
    def is_equal_class(self, class_id: int) -> bool:
        """
        Check if a class ID is part of an equal class group.
        
        Args:
            class_id: The class ID to check
            
        Returns:
            True if the class is part of an equal class group, False otherwise
        """
        return class_id in self.equal_class_lookup
    
    def get_all_equal_classes(self) -> Set[int]:
        """
        Get all class IDs that are part of equal class groups.
        
        Returns:
            Set of all class IDs in equal class groups
        """
        return set(self.equal_class_lookup.keys())
    
    def get_all_hier_classes(self) -> Dict[str, List[int]]:
        """
        Get all hierarchical class mappings.
        
        Returns:
            Dictionary of hierarchical class mappings
        """
        return self.hier_classes.copy()
    
    def is_in_equal_group_with_any(self, candidate_ids: List[int], class_id: int) -> bool:
        """
        Check whether a class ID shares an equal group with any ID from the provided list.
        
        Args:
            class_id: The class ID to check membership for.
            candidate_ids: List of class IDs to compare against.
        
        Returns:
            True if the class is in the same equal group as any of the candidate IDs
            (or if any candidate equals the class_id). False otherwise.
        """
        if not candidate_ids:
            return False
        
        candidates_set = set(candidate_ids)
        
        # Trivial match
        if class_id in candidates_set:
            return True
        
        neighbors = self.get_neighbors(class_id)
        if not neighbors:
            return False
        
        return len(neighbors.intersection(candidates_set)) > 0
    
    def get_image_overrides(self) -> Dict[str, str]:
        """
        Get all image-specific annotator overrides.
        
        Returns:
            Dictionary mapping image_name -> annotator_name
        """
        return self.image_overrides.copy()
    
    def get_image_override(self, image_name: str) -> Union[str, None]:
        """
        Get the annotator override for a specific image.
        
        Args:
            image_name: The image name/ID
            
        Returns:
            Annotator name if override exists, None otherwise
        """
        return self.image_overrides.get(image_name)