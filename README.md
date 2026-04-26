# Aiming for Perfect ImageNet-1K

Research hub for our ImageNet-1K reannotation and benchmarking efforts —
[Visual Recognition Group](https://vrg.fel.cvut.cz), Czech Technical University in Prague.

**Live site:** https://klarajanouskova.github.io/ImageNet/

## Site structure

| Path | Description |
|------|-------------|
| `index.html` | Hub page (plain HTML, mustard yellow theme) |
| `blog/` | Blog post source (Jekyll / al-folio / Distill) |
| `.github/workflows/deploy.yml` | Builds blog → `_deploy/blog/`, copies hub, deploys via GitHub Pages Actions |

The hub deploys at `/ImageNet/` and the blog at `/ImageNet/blog/`.

## Local development

### Hub page only
Open `index.html` directly in your browser — no build step needed.

### Full site (hub + blog)

```bash
# One-time: install Ruby gems
cd blog && bundle install && cd ..

# Build and serve (from repo root)
mkdir -p _deploy/blog && \
cd blog && bundle exec jekyll build \
  --config _config.yml,_config_local.yml \
  --destination ../_deploy/blog && cd .. && \
cp index.html _deploy/index.html && \
python3 -m http.server 8000 --directory _deploy
```

Then open:
- Hub → http://localhost:8000/
- Blog → http://localhost:8000/blog/

`_config_local.yml` overrides `baseurl` to `/blog` for local serving.
In CI, only `_config.yml` is used (`baseurl: /ImageNet/blog`).

### Deploying

1. Push the `hub` branch: `git push -u origin hub`
2. In repo Settings → Pages → set Source to **GitHub Actions**

The workflow triggers automatically on every push to `hub`.

---

## Repository contents

## Quick Overview of Contents:
- `eval_corrections/load_data/` - scripts for loading existing ImageNet corrections.
- `eval_corrections/verify_images/` - scripts for evaluating corrections.
    - `eval_corrections/verify_images/results/clean_validation.csv` - clean validation set, obtained by combining existing corrections.

- `expert_annotations/356_357_358_359.json` - expert annotations for ImageNet classes `356`, `357`, `358`, and `359` (weasel-like family).

- `classes/modified_classnames.txt` - set of modified class names, built on [OpenAI’s version](https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
- `classes/problem_groups/CATEGORIES.md` - list of problematic categories.
- `classes/problem_groups/clusters.json` - list of problematic groups containing the assignment of ImageNet classes to defined categories.