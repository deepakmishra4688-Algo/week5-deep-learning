# Week 5 Task — Deep Learning Application in Data Science

**Dataset:** UCI Adult / Census Income (same cleaned dataset and engineered features as Week 4, for a fair comparison)
**Problem:** Binary classification — predict income >$50K vs <=$50K
**Framework:** PyTorch
**Source:** https://archive.ics.uci.edu/dataset/2/adult
**Citation:** Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5XW20

## Contents

- `Week5_Deep_Learning_Report.docx` — the full write-up: architecture design and justification, training process, overfitting ablation study, evaluation, and critical comparison against Week 4's classical models.
- `Week5_Deep_Learning.ipynb` — the executable notebook the report's numbers and charts were generated from.
- `adult_dataset_cleaned.csv` — the cleaned input dataset (output of the Week 1 task).
- `outputs/` — the 6 charts referenced in the report, including the architecture diagram.

## How to reproduce

1. Place `adult_dataset_cleaned.csv` (Week 1 output) in the working directory.
2. Run `Week5_Deep_Learning.ipynb` top to bottom.
3. All 6 charts in `outputs/` will be regenerated. Training takes ~12 seconds on CPU.

## Architecture summary

- **Model:** Feedforward MLP — Linear(48→64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64→32) → BatchNorm → ReLU → Dropout(0.3) → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid
- **Parameters:** 5,953 total
- **Loss:** BCEWithLogitsLoss with `pos_weight=3.18` (handles the ~76/24 class imbalance, same intent as `class_weight="balanced"` in Week 4)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5) with ReduceLROnPlateau scheduler
- **Training:** early stopping (patience=8), stopped at epoch 25/60

## Key results

| Metric | Logistic Regression (Wk 4) | Random Forest, tuned (Wk 4) | MLP (this week) |
|---|---|---|---|
| Accuracy | 81.23% | 82.41% | 80.56% |
| Precision | 57.27% | 59.09% | 56.02% |
| Recall | 85.02% | 86.26% | 87.50% |
| F1 | 68.44% | 70.14% | 68.30% |
| ROC-AUC | 91.25% | 92.45% | 91.60% |

- The MLP lands between Logistic Regression and the tuned Random Forest — it does **not** outperform the classical models, which the report discusses honestly as an expected outcome on structured tabular data with engineered features, rather than a shortcoming to hide.
- **Overfitting ablation:** the same architecture trained with vs without Dropout/BatchNorm shows a clear difference — the unregularized version's train/validation loss gap (0.072) is roughly 6x larger than the regularized version's gap (0.012), with the unregularized validation loss visibly rising after ~epoch 15 while train loss keeps falling.
