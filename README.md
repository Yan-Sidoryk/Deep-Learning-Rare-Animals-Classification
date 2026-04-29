# Rare Species Classification
### Deep Learning Group Project — Bachelor's Degree in Data Science 2025/26

---

## Overview

This project develops a deep learning model to classify rare species by their **biological family** from images. The dataset is sourced from the Encyclopedia of Life (EOL) and was curated as part of the [BioCLIP](https://arxiv.org/abs/2311.18803) study. Each image is paired with taxonomic metadata (kingdom, phylum, family), and the goal is to predict the family label from the image alone.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Approach

We explored multiple architectures and preprocessing strategies, using validation performance to select our best model. Data was split into training, validation, and test sets — the test set was kept fully held out until final evaluation.

---

## Results

| Model | Val Accuracy | Notes |
|-------|-------------|-------|
| Baseline CNN | 2.5% | Trained from scratch |
|EfficientNetB0 | 63% | Transfer learning |
| Fine-tuned pretrained | 84% | **Best model** |

Final project grade - 17/20

---

## Reference

Stevens et al. (2024). *BioCLIP: A Vision Foundation Model for the Tree of Life*. CVPR 2024.
