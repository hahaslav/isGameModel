# Detect if a game is shown on screenshot

There are three models:
- [Logistical Regression](simple_model.py)
- [CNN](mid_model.py)
- [Transformer](transformer_model.py) ([fine-tune](transformer_training.ipynb) of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k))

The scripts allow for model training and inference. There is also a live mode to run model on desktop content in real time.

The models were trained on data taken with [my previous project](https://github.com/hahaslav/selfstalk).

## Usage

Install [uv](https://docs.astral.sh/uv/#installation) to sync dependencies. To run marimo notebooks, run:
```
uv run marimo edit notebook.py
```
