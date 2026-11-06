# ReImageNet: Doomed to Re-Annotate, Forever: The ImageNet Story

Code and analysis for the paper *Doomed to Re-Annotate, Forever: The ImageNet Story*
(NeurIPS 2026, Evaluations & Datasets Track).
[Visual Recognition Group](https://vrg.fel.cvut.cz), Czech Technical University in Prague.

- Paper: `<FILL link>`
- Annotations: https://huggingface.co/datasets/vrg-prague/ReImageNet
- Annotation preview: https://vrg.fel.cvut.cz/reimagenet
- Project hub / blog: https://klarajanouskova.github.io/ImageNet/

## What is in this repository

| Path | Description |
|------|-------------|
| `eval_reimagenet/` | Loading ReImageNet, evaluation of models with original and ReImageNet labels, label statistics |
| `<FILL>` | Derivative benchmark analysis (Sec. 5.3) |

## Setup

```bash
git clone https://github.com/klarajanouskova/ImageNet
cd ImageNet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The evaluation code needs Python 3.9 or newer and only the standard library.
`huggingface_hub` is needed to download the annotations.

You also need the ImageNet-1k (ILSVRC2012) validation images to run models, which we do not redistribute.
Download them from https://www.image-net.org/ (subject to the ImageNet terms of use).
The evaluation itself only needs prediction files.

## Getting the annotations

```bash
python -m eval_reimagenet.download                 # latest revision, into data/reimagenet/
python -m eval_reimagenet.download --revision f0bcc5f82c0b6011cfe833590bce9019484c20f1
```

The files go to `data/reimagenet/` by default. Set `REIMAGENET_DIR` to use another directory.
The dataset commit hash is written to `data/reimagenet/REVISION`. The paper uses revision `<FILL>`.

The release has three files:

| File | Content |
|------|---------|
| `reannotation.jsonl` | One record per validation image: `image_name`, `original_class` (list), `reannotated_labels` (list, `-1` = not sure), `file_path` (`{synset}/{image_name}`), `bboxes` |
| `label_names.json` | Synset ids indexed by class id (0-999) |
| `class_update_config.json` | `eq_classes`: pairs of visually indistinguishable classes that are treated as one class in evaluation |

Each box has `coordinates` (`[x1, y1, x2, y2]` in pixels), `labels`, `group`, and the attributes
`crowd_flag`, `ocr_needed_flag` (text recognition), `rendition_flag`, `reflected_flag` and `dominant_object`.

```python
from eval_reimagenet import load_reimagenet, group_images

annotations, eq_classes, label_names = load_reimagenet()   # data/reimagenet/ or $REIMAGENET_DIR
groups = group_images(annotations, eq_classes)              # {"N": [...], "S+": [...], "S-": [...], "M+": [...], "M-": [...]}
```

## Image categories and metric

Let `alpha_i` be the set of valid ReImageNet classes of image `i` ("not sure" removed, an eq-class pair counted as one class)
and `y_i` its original ImageNet label.

- `N`: no valid ImageNet-1k class. These images are excluded from evaluation.
- `S`: exactly one class. `S+` if `y_i` is that class, `S-` otherwise.
- `M`: more than one class. `M+` if `y_i` is one of them, `M-` otherwise.
- `A\N`: all images except `N`.

A top-1 prediction is correct with the original labels (GT) if it is `y_i`,
and with ReImageNet labels (ReGT) if it is in `alpha_i`. Both treat the classes of an eq-class pair as the same class.

## Reproducing the results

```bash
# 1. Top-1 predictions of a model on ImageNet-1k val (format below)
#    <FILL: script or released prediction files>

# 2. Top-1 accuracy with original vs. ReImageNet labels (Table <FILL>, main results)
python -m eval_reimagenet.evaluate --preds preds/eva02.csv preds/gpt54.json --names EVA-02 GPT-5.4
python -m eval_reimagenet.evaluate --preds preds/dinov3_knn.csv --pred-column k_11_pred   # DINOv3 11-NN
python -m eval_reimagenet.evaluate --preds preds/*.csv --format latex                    # or --format csv

# 3. Label statistics (12.2% incorrect original label, 31.9% multilabel, 4.0% no ImageNet-1k object)
python -m eval_reimagenet.label_stats

# 4. Derivative benchmarks (Sec. 5.3)
# <FILL>
```

`evaluate` prints GT accuracy on `A\N` and ReGT accuracy on `A\N`, `S`, `S+`, `S-`, `M`, `M+`, `M-`.
It fails if an evaluated image has no prediction. Use `--allow-missing` to evaluate only the predicted images.

Prediction files map validation image names to ImageNet-1k class ids (`-1` = no prediction):

```text
# CSV: an img_id column and a prediction column (top_1_pred by default, --pred-column to change)
img_id,top_1_pred
ILSVRC2012_val_00000001.JPEG,65

# JSON: a class id, or {"label": id, ...}, per image, optionally under "predictions"
{"predictions": {"ILSVRC2012_val_00000001.JPEG": {"label": 65, "similarity": 0.957}}}
```

Free-text answers of MLLMs must be mapped to class ids first. In the paper, they are mapped with
text embeddings to the closest class name. `<FILL: script for this step>`

Closed-source models (GPT-5.4) need `OPENAI_API_KEY`, and their outputs are not exactly reproducible.
Open-weight models and exact versions are listed in the paper's supplementary (Sec. "Experimental setup").

## License

- Code: `<FILL: MIT / Apache-2.0>`
- Annotations: `<FILL: must match the Hugging Face card and Croissant file>`
- ImageNet images are not included and remain under the ImageNet terms of use.

## Citation

```bibtex
@inproceedings{volkov2026reimagenet,
  title     = {Doomed to Re-Annotate, Forever: The ImageNet Story},
  author    = {Volkov, Illia and Kisel, Nikita and Mishkina, Tetiana and Janouskova, Klara and Matas, Jiri},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
