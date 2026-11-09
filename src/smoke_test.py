"""Quick sanity check that the ML stack imports and works."""
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib

print(f"numpy       {np.__version__}")
print(f"pandas      {pd.__version__}")
print(f"scikit-learn{sklearn.__version__}")
print(f"xgboost     {xgb.__version__}")
print(f"lightgbm    {lgb.__version__}")
print(f"shap        {shap.__version__}")

# Tiny end-to-end test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
model = LogisticRegression().fit(X, y)
auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
print(f"\nSmoke test ROC-AUC: {auc:.3f}  (should be > 0.8)")