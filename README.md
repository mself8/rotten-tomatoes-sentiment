# Rotten Tomatoes Phrase Sentiment (5-class)

Course final project (Deep Learning, University of Seoul, Spring 2026): 5-way sentiment classification of Rotten Tomatoes movie-review **phrases** (Kaggle-style class competition), implemented **from scratch in pure PyTorch** — no pretrained models beyond GloVe word vectors.

## Approach

- **Model**: GloVe embeddings → BiLSTM → attention pooling (many-to-one), as a **4-model ensemble** varying embedding (6B / 840B), capacity, and pooling — each trained with 5-fold CV
- **Sentence-level GroupKFold**: all 156,060 training phrases are sub-spans of just 8,529 source sentences, so folds are split by sentence to prevent leakage between near-identical phrases
- **Exact-match post-processing**: test phrases that appear verbatim in the training set take their training label (uses training labels only — competition-rule compliant)

## Results (Kaggle public leaderboard)

| Configuration | Accuracy |
|---|---|
| Majority-class baseline (neutral) | 0.510 |
| 4-model ensemble | 0.681 |
| + exact-match post-processing | **0.706** |

Sentence-level out-of-fold validation accuracy: ~0.661. The pipeline was re-run end-to-end and verified to reproduce the submitted predictions bit-for-bit.

## Reproducing

Everything lives in **`Final.ipynb`** (data prep → EDA → training → ensembling → post-processing → submission file). Run it top to bottom.

Not included — place these in the repo root first:

- `train.tsv` / `test.tsv` — the course Kaggle competition data (the classic *Sentiment Analysis on Movie Reviews* format)
- [GloVe vectors](https://nlp.stanford.edu/projects/glove/): `glove.6B` and `glove.840B.300d`

Dependencies: PyTorch, numpy, pandas, scikit-learn. A GPU is recommended (4 models × 5 folds).
