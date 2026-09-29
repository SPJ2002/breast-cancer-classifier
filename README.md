# Breast Cancer Diagnosis Classifier

Predicts whether a breast tumour is **malignant or benign** from measurements of cell nuclei, using the Breast Cancer Wisconsin (Diagnostic) dataset (569 samples, 30 features; 212 malignant, 357 benign).

## What it does
1. Loads the dataset from Scikit-Learn and explores class balance and feature correlations.
2. Splits the data 80/20 with stratification (random_state=42).
3. Compares Logistic Regression, Random Forest and SVM using 5-fold stratified cross-validation (scaling is done inside a pipeline, so there is no data leakage).
4. Tunes the best model (by cross-validated ROC-AUC) with GridSearchCV.
5. Evaluates once on the held-out test set and plots the confusion matrix, ROC curve and permutation feature importance.

## Results
| Model | CV accuracy | CV recall | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.974 | 0.953 | 0.996 |
| Random Forest | 0.965 | 0.941 | 0.989 |
| SVM | 0.971 | 0.947 | 0.995 |

Tuned Logistic Regression (C=1) on the held-out test set: **accuracy 0.965, precision 0.975, recall 0.929, ROC-AUC 0.996**.

Plots are in `results/`. Most influential features: worst symmetry, worst texture, worst concavity, worst concave points, mean concave points.

## Why recall matters
Malignant is the positive class. A missed malignant case (false negative) is far more costly than a false alarm, so I report recall alongside accuracy. The model missed some malignant cases on the test set (recall 0.929), so a real screening tool would likely lower the decision threshold to trade precision for recall.

## Limitations and next steps
- Small, clean, well-studied dataset. This is a learning project, not a clinical tool.
- Model selection used one train/test split; repeated CV or a second dataset would give a more reliable estimate.
- Next: threshold tuning for higher recall, and calibration of predicted probabilities.

## Run it
```bash
pip install -r requirements.txt
python classifier.py
```
