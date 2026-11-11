# src/preprocessing.py — corrected to auto-discover categoricals
"""Preprocessing pipeline for S6E11 Credit Card Fraud."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

TARGET = "is_fraud"
ID_COL = "id"

SKEWED_COLS = ["amount_usd", "distance_from_home_km", "account_balance_usd"]
# NOTE: no more CAT_COLS constant — discovered from data instead


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)

    for col in SKEWED_COLS:
        if col in cols:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    if {"velocity_score", "hours_since_last_txn"}.issubset(cols):
        df["velocity_per_hour"] = df["velocity_score"] / (df["hours_since_last_txn"] + 1)

    if {"cvv_retry_count", "prior_disputes"}.issubset(cols):
        df["risk_stack"] = df["cvv_retry_count"] + df["prior_disputes"]

    if {"txn_count_last_24h", "amount_usd"}.issubset(cols):
        df["amount_per_txn_24h"] = df["amount_usd"] / (df["txn_count_last_24h"] + 1)

    return df


def get_feature_lists(df: pd.DataFrame):
    """Discover categoricals from dtype — no hardcoded assumptions.

    Any column whose dtype is object / string / category is treated as
    categorical. Everything else numeric. This catches card_network,
    transaction_channel, and any other string column we haven't seen yet.
    """
    all_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]

    cat_cols = (
        df[all_cols]
        .select_dtypes(include=["object", "string", "category"])
        .columns
        .tolist()
    )
    num_cols = [c for c in all_cols if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", numeric_pipe, num_cols)]
    if cat_cols:  # only add if there are actually categoricals
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers)


def get_splits(test_size=0.2, random_state=42):
    train, test = load_data()
    train = add_features(train)
    test  = add_features(test)

    X = train.drop(columns=[TARGET, ID_COL])
    y = train[TARGET].astype(int)
    X_test = test.drop(columns=[ID_COL])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_val, y_train, y_val, X_test, test[ID_COL]


if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val, X_te, ids = get_splits()
    num_cols, cat_cols = get_feature_lists(X_tr)

    print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
    print(f"Fraud in train: {y_tr.mean()*100:.3f}%  in val: {y_val.mean()*100:.3f}%")
    print(f"\nNumeric features ({len(num_cols)}):")
    print("  " + ", ".join(num_cols))
    print(f"\nCategorical features ({len(cat_cols)}):")
    print("  " + ", ".join(cat_cols))