# Peptide Detectability Prediction (Bidirectional GRU)

A deep learning model that predicts whether a peptide is **detectable by mass spectrometry**
directly from its amino-acid sequence. This is framed as a binary classification task
(`0 = not detectable`, `1 = detectable`).

## Model architecture

Two stacked **Bidirectional GRU** layers (encoder → decoder):

```
Input (40 × 21 one-hot)
      │
   Masking                        (ignore zero-padding)
      │
BiGRU(64, return_sequences)       ← context in both directions
      │
BiGRU(64)                         ← collapse to a sequence vector
      │
Dropout → Flatten
      │
Dense(128, ReLU) → Dropout
      │
Dense(1, sigmoid)                 ← detectability probability
```

## Data

Public peptide-detectability datasets from the [Wilhelm Lab](https://huggingface.co/Wilhelmlab)
on Hugging Face:

| Dataset | Role |
|---|---|
| `detectability-proteometools` | Train / validation / internal test |
| `detectability-sinitcyn` | Train / validation / internal test |
| `detectability-wang` | **External** hold-out test set |

ProteomeTools + Sinitcyn are pooled and re-split (stratified 80/20/…); the Wang set is held
out entirely as an independent generalization benchmark.

## Usage

```bash
pip install -r requirements.txt
jupyter lab peptide_detectability_bigru.ipynb
```

Run the cells top to bottom. Data is streamed from Hugging Face, so an internet connection
is required on first run.

> **Tip:** Training is much faster on a GPU. The recurrent (GRU) layers are the bottleneck —
> on CPU each epoch takes several minutes, whereas a CUDA-enabled GPU is typically an order of
> magnitude faster. Check that one is visible with `tf.config.list_physical_devices('GPU')`
> before training.

## Outputs

- `peptide_alldata_model_weights_detectability.*` — best model weights (by val accuracy)
- `training_history.png` — train/validation loss & accuracy curves
- `confusion_matrix_wang.png` — confusion matrix on the external test set
- `roc_pr_curves_wang.png` — ROC and precision–recall curves

## Metrics reported

Classification report, confusion matrix, ROC-AUC, average precision, and
sensitivity / specificity / FPR / FNR on the external Wang test set.
