"""Breast Cancer Diagnosis Classifier.

Compares Logistic Regression, Random Forest and SVM on the Breast Cancer
Wisconsin (Diagnostic) dataset, tunes the best model, and evaluates it on
held-out data. Malignant is treated as the positive class (label 1).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
OUT = "results"

# 1. Load data. sklearn labels malignant as 0, so flip it: malignant = 1.
data = load_breast_cancer(as_frame=True)
X = data.data
y = 1 - data.target
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
print("Class balance (0 = benign, 1 = malignant):")
print(y.value_counts().to_string(), "\n")

# 2. Quick EDA plots
plt.figure(figsize=(4, 3))
sns.countplot(x=y.map({0: "benign", 1: "malignant"}))
plt.title("Class balance"); plt.tight_layout()
plt.savefig(f"{OUT}/class_balance.png", dpi=150); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(X.iloc[:, :10].corr(), cmap="coolwarm", center=0)
plt.title("Correlation of the 10 mean features"); plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150); plt.close()

# 3. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# 4. Compare models with 5-fold stratified cross-validation
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Random Forest": Pipeline([("scale", StandardScaler()),
                               ("clf", RandomForestClassifier(random_state=SEED))]),
    "SVM": Pipeline([("scale", StandardScaler()),
                     ("clf", SVC(probability=True, random_state=SEED))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for name, model in models.items():
    s = cross_validate(model, X_train, y_train, cv=cv,
                       scoring=["accuracy", "recall", "roc_auc"])
    cv_results[name] = {m: float(s[f"test_{m}"].mean()) for m in ["accuracy", "recall", "roc_auc"]}
    print(f"{name:20s} CV acc={cv_results[name]['accuracy']:.3f} "
          f"recall={cv_results[name]['recall']:.3f} AUC={cv_results[name]['roc_auc']:.3f}")

best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"])
print(f"\nBest by CV ROC-AUC: {best_name}")

# 5. Tune the best model with GridSearchCV
grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10]},
    "SVM": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]},
}
search = GridSearchCV(models[best_name], grids[best_name], cv=cv, scoring="roc_auc")
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_, f"(CV AUC={search.best_score_:.3f})")

# 6. Evaluate on held-out test set
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
test = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("\nHeld-out test set:")
for k, v in test.items():
    print(f"  {k}: {v:.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["benign", "malignant"])
plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150); plt.close()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title(f"ROC curve - {best_name}"); plt.tight_layout()
plt.savefig(f"{OUT}/roc_curve.png", dpi=150); plt.close()

# 7. Feature importance (permutation, works for any model)
imp = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc")
top = pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(10)
top.plot.barh(figsize=(7, 4)); plt.title("Top 10 features (permutation importance)")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()
print("\nTop features:", ", ".join(top.index[::-1][:5]))

json.dump({"cv": cv_results, "best_model": best_name, "best_params": search.best_params_,
           "test": test, "top_features": list(top.index[::-1])},
          open(f"{OUT}/results.json", "w"), indent=2, default=float)
