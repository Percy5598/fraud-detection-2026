"""Train baseline and LightGBM on S6E11 fraud data."""
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    roc_curve,
)
from preprocessing import (
    get_splits, get_feature_lists, build_preprocessor,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def recall_at_fpr(y_true, y_proba, fpr_target=0.01):
    """Recall achievable while keeping FPR ≤ fpr_target (production-relevant metric)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    idx = np.searchsorted(fpr, fpr_target)
    return tpr[min(idx, len(tpr) - 1)]


def evaluate(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "model":         name,
        "threshold":     threshold,
        "roc_auc":       roc_auc_score(y_true, y_proba),
        "pr_auc":        average_precision_score(y_true, y_proba),
        "recall@fpr1%":  recall_at_fpr(y_true, y_proba, 0.01),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n{'─' * 50}")
    print(f"{name}  (threshold={threshold:.2f})")
    print(f"{'─' * 50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15} {v:.4f}")
    return metrics


def train_baseline():
    t0 = time.time()
    X_tr, X_val, y_tr, y_val, _, _ = get_splits()
    num_cols, cat_cols = get_feature_lists(X_tr)
    print(f"Features — numeric: {len(num_cols)}, categorical: {len(cat_cols)}")
    print(f"Fraud rate — train: {y_tr.mean()*100:.3f}%  val: {y_val.mean()*100:.3f}%")

    pipe = Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols)),
        ("clf",  LogisticRegression(
            class_weight="balanced",
            solver="saga",           # fast on large data
            penalty="l2",
            C=1.0,
            max_iter=200,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    print("\nFitting LogReg baseline (saga solver, ~1-3 min)...")
    pipe.fit(X_tr, y_tr)
    print(f"Fit time: {time.time() - t0:.1f}s")

    proba = pipe.predict_proba(X_val)[:, 1]
    metrics = evaluate("LogReg baseline", y_val, proba, threshold=0.5)

    joblib.dump(pipe, MODELS_DIR / "logreg_baseline.pkl")
    print(f"\nModel saved → {MODELS_DIR / 'logreg_baseline.pkl'}")
    return pipe, y_val, proba, metrics


if __name__ == "__main__":
    train_baseline()