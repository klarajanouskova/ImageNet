"""Download the ReImageNet annotations from Hugging Face.

Fetches ``reannotation.jsonl``, ``label_names.json``, ``class_update_config.json`` and the
dataset card into ``data/reimagenet/`` (or ``--out-dir`` / ``$REIMAGENET_DIR``), and writes
the dataset commit hash to ``REVISION`` so results can be tied to a dataset version.

Usage:
    python -m eval_reimagenet.download
    python -m eval_reimagenet.download --revision f0bcc5f82c0b6011cfe833590bce9019484c20f1
"""

import argparse
import json
from pathlib import Path

from .annotations import ANNOTATIONS_FILE, CLASS_CONFIG_FILE, DEFAULT_DATA_DIR, LABEL_NAMES_FILE

REPO_ID = "vrg-prague/ReImageNet"
FILES = [ANNOTATIONS_FILE, CLASS_CONFIG_FILE, LABEL_NAMES_FILE, "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default="main", help="Branch, tag or commit hash (default: main)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    for name in FILES:
        path = hf_hub_download(REPO_ID, name, repo_type="dataset", revision=sha, local_dir=args.out_dir)
        print(f"downloaded {path}")
    (args.out_dir / "REVISION").write_text(sha + "\n")

    with open(args.out_dir / CLASS_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.out_dir / ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        n_records = sum(1 for line in f if line.strip())
    print(f"\nReImageNet @ {sha}")
    print(f"  {n_records} image records")
    print(f"  {len(config.get('eq_classes', {}))} eq-class groups")


if __name__ == "__main__":
    main()
