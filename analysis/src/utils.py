"""
Utility functions for the reannotation experiments project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('reannotation_experiments.log')
        ]
    )


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise


def extract_username_from_filename(filename: str) -> str:
    """Extract username from checkbox_selections_{username}.json filename."""
    if filename.startswith("checkbox_selections_") and filename.endswith(".json"):
        return filename[len("checkbox_selections_"):-len(".json")]
    else:
        raise ValueError(f"Invalid filename format: {filename}")


def has_annotator_suffix(filename: str) -> bool:
    """Check if filename has _S or _M suffix (e.g., checkbox_selections_{username}_S.json)."""
    if filename.startswith("checkbox_selections_") and filename.endswith(".json"):
        username_part = filename[len("checkbox_selections_"):-len(".json")]
        return username_part.endswith("_S") or username_part.endswith("_M")
    return False