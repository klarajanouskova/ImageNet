# Doomed to Re-Annotate, Forever: The ImageNet Story

Code for the NeurIPS paper of the same name.

## Structure

| Directory | Purpose |
|-----------|---------|
| `data/` | Raw annotations, processed outputs, embeddings, distribution plots |
| `inference/` | Model inference pipelines and scripts (not yet publicly released) |
| `analysis/` | Accuracy calculation and all paper experiments |
| `config/` | Paths and dataset roots — see [config/settings.py](config/settings.py) |

> **Note:** Data files are not included in this repo and must be downloaded separately (see `config/settings.py` for expected paths).

## Reproducing experiments

All experiments from the paper (including supplementary) are run through a single entry point:

```
python analysis/run_analysis.py [options]
```

Pass `-p <predictions_file>` to point at model outputs (CSV or JSON); use `--experiment <name>` to select a specific experiment (default is E1, overall accuracy). Run with `--help` or see the docstring at the top of [analysis/run_analysis.py](analysis/run_analysis.py) for the full list of experiments (E1–E10) and their flags.
