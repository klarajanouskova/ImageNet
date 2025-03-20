# Flaws of ImageNet
This repository provides supplementary material for our ICLR 2025 blog post 'Flaw's of ImageNet, Computer Vision's Favourite Dataset'.

## Quick Overview of Contents:
- `eval_corrections/load_data/` - scripts for loading existing ImageNet corrections.
- `eval_corrections/verify_images/` - scripts for evaluating corrections.
    - `eval_corrections/verify_images/results/clean_validation.csv` - clean validation set, obtained by combining existing corrections.

- `expert_annotations/356_357_358_359.json` - expert annotations for ImageNet classes `356`, `357`, `358`, and `359` (weasel-like family).

- `classes/modified_classnames.txt` - set of modified class names, built on [OpenAI’s version](https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
- `classes/problem_groups/CATEGORIES.md` - list of problematic categories.
- `classes/problem_groups/clusters.json` - list of problematic groups containing the assignment of ImageNet classes to defined categories.