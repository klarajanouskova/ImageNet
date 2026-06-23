"""
Configuration settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MAPPINGS_DIR = DATA_DIR / "mappings"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CSV_PREDICTIONS_DIR = PREDICTIONS_DIR / "csv"
CROP_DIR = PREDICTIONS_DIR / "crop"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
CORRECTIONS_DIR = DATA_DIR / "corrections"
PLOTS_DIR = DATA_DIR / "plots"
PROCESSED_DATA_DIR = DATA_DIR / "results"  # runtime-generated outputs (evaluation reports, cache)

# Google Drive API settings
TOKEN_FILE = PROJECT_ROOT / "token.json"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/spreadsheets.readonly',
]

# File patterns
ANNOTATION_FILE_PATTERN = "checkbox_selections_{username}.json"

# ImageNet validation folder.
# Set the IMAGENET_VAL_PATH environment variable to override the default.
# On RCI cluster:  export IMAGENET_VAL_PATH=/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val
# Locally:         export IMAGENET_VAL_PATH=/path/to/your/imagenet/val
ANNOTATIONS_ROOT_FOLDER = os.environ.get(
    "IMAGENET_VAL_PATH",
    "/Users/gonikisgo/val"  # fallback for local dev
)

# Google Drive folder IDs to download from
FOLDER_IDS = [
    "1RvuEVaIxRSRfGbIO3B2AGH1YtAEmRzIb",
    "17QieZdZUxCF-154SRdX9mtRqH8D2wj7B",
    "1OeSdT3fgUpTzZZxx6Z4fiolK6bPqUDyl",
    "1P9Ytjb-F8lFNVDuRB3OTLDqNlFnnICZ2",
    "1pfja6k8fDldME0_NpRys8umNtvLfXtK9",
    "1Ab58vpbq9BthHGKmpasEAFXge88npOje",
    "1cLs9i0LiEA8M9kMvbX3QYuHuosoMJvgk",
    "1KwDjjWdSboq5OcNsPBOdm7V_WCRD2mag"
]

FOLDER_SP_IDS = [
    "1RvuEVaIxRSRfGbIO3B2AGH1YtAEmRzIb",
    "1Ab58vpbq9BthHGKmpasEAFXge88npOje",
    "1KwDjjWdSboq5OcNsPBOdm7V_WCRD2mag"
]
