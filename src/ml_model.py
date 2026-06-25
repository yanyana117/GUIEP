"""
Seismic event classifier: natural earthquake vs man-made explosion (GUIEP)
=========================================================================

An end-to-end supervised-learning pipeline on the USGS catalogue that mirrors a
real data-science workflow:

    1. COLLECT DATA VIA SQL  - features are pulled from the SQLite database
                               (built by ``sql_analysis.py``) with a SQL query.
    2. EXPLORATORY ANALYSIS  - quantify how the two classes separate.
    3. MODEL (scikit-learn)  - train and evaluate a classifier with proper
                               train/test split, cross-validation, and metrics
                               suited to an imbalanced target.

The task is the classic seismic **discrimination problem** (used in nuclear
test monitoring / CTBT verification): given an event's location, depth, time and
magnitude, is it a natural earthquake or a man-made explosion? Man-made events
in this catalogue are almost all Nevada Test Site nuclear tests - they are very
shallow (median depth 0 km vs 7.2 km for earthquakes), clustered in Nevada, and
confined to 1962-1999, so the signal is real and the model is interpretable.

Usage
-----
    python src/ml_model.py                 # train + evaluate, save figure
    python src/ml_model.py --plot none     # skip the figure

Target is imbalanced (~7.6% man-made), so we report precision/recall/F1/ROC-AUC,
not just accuracy, and use class-balanced weights.
"""
import argparse
import os

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the exact same loader + DB builder as the SQL module.
from sql_analysis import build_database, load_dataframe

FEATURES = ["latitude", "longitude", "depth", "mag", "year"]
TARGET = "is_manmade"


# --------------------------------------------------------------------------- #
#  1. Collect the modelling data with a SQL query
# --------------------------------------------------------------------------- #
def collect_features(conn):
    """Pull the feature matrix straight out of SQLite with SQL.

    The label is derived in SQL: anything that is not a natural 'earthquake'
    (nuclear explosion, quarry blast, mine collapse, ...) is man-made.
    """
    sql = """
        SELECT latitude,
               longitude,
               depth,
               mag,
               year,
               CASE WHEN type = 'earthquake' THEN 0 ELSE 1 END AS is_manmade
        FROM   earthquakes;
    """
    return pd.read_sql_query(sql, conn)


# --------------------------------------------------------------------------- #
#  2. Exploratory analysis - show that the classes actually separate
# --------------------------------------------------------------------------- #
def explore(data):
    n = len(data)
    pos = int(data[TARGET].sum())
    print(f"Rows: {n}   earthquakes: {n - pos}   man-made: {pos} "
          f"({100 * pos / n:.1f}%)")
    print("\nMedian feature value by class (the signal the model will use):")
    summary = (data.groupby(TARGET)[["depth", "mag", "year", "latitude"]]
               .median().round(2))
    summary.index = ["earthquake", "man-made"]
    print(summary.to_string())
    print("\nDepth is the giveaway: man-made events are surface blasts "
          "(~0 km) while quakes nucleate kilometres down.")


# --------------------------------------------------------------------------- #
#  3. Train + evaluate scikit-learn models
# --------------------------------------------------------------------------- #
def build_models():
    """A baseline plus two real classifiers, each as an imputing pipeline."""
    imp = ("imputer", SimpleImputer(strategy="median"))  # depth is ~18% missing
    return {
        "baseline (most frequent)": Pipeline([
            imp, ("clf", DummyClassifier(strategy="most_frequent"))]),
        "logistic regression": Pipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000,
                                       class_weight="balanced"))]),
        "random forest": Pipeline([
            imp, ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42))]),
    }


def evaluate(data, plot_path):
    X = data[FEATURES]
    y = data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best = None
    for name, model in build_models().items():
        # Cross-validated ROC-AUC on the training split (robust to imbalance).
        if name.startswith("baseline"):
            cv_auc = float("nan")
        else:
            cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv,
                                     scoring="roc_auc").mean()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = (model.predict_proba(X_te)[:, 1]
                 if hasattr(model, "predict_proba") else pred)
        test_auc = roc_auc_score(y_te, proba)

        print(f"\n=== {name} ===")
        if not np.isnan(cv_auc):
            print(f"5-fold CV ROC-AUC (train): {cv_auc:.3f}")
        print(f"Hold-out ROC-AUC (test)  : {test_auc:.3f}")
        print(classification_report(y_te, pred,
              target_names=["earthquake", "man-made"], zero_division=0))
        if name == "random forest":
            print("Confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(y_te, pred))
            best = model

    if plot_path and plot_path.lower() != "none" and best is not None:
        plot_importance(best, plot_path)


def plot_importance(rf_pipeline, out_path):
    """Bar chart of random-forest feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = rf_pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURES[i] for i in order], importances[order], color="#c44e52")
    ax.set_title("What tells a blast from a quake (random-forest importance)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved feature-importance plot: {os.path.relpath(out_path)}")


# --------------------------------------------------------------------------- #
#  4. Driver
# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(__file__)
    default_data = os.path.join(here, "..", "data",
                                "usgs_earthquakes_1900_2020.csv")
    default_plot = os.path.join(here, "..", "results", "feature_importance.png")
    p = argparse.ArgumentParser(description="Earthquake vs man-made classifier")
    p.add_argument("--data", default=default_data, help="path to USGS CSV")
    p.add_argument("--plot", default=default_plot,
                   help="feature-importance figure path, or 'none'")
    args = p.parse_args()

    df = load_dataframe(args.data)
    conn = build_database(df)           # CSV -> SQLite (in-memory)
    data = collect_features(conn)       # features pulled back out via SQL
    conn.close()

    explore(data)
    evaluate(data, args.plot)


if __name__ == "__main__":
    main()
