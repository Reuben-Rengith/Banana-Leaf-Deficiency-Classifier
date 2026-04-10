from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold


def train_meta_gbm(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    results_dir: str,
    cv_splits: int = 5,
    seed: int = 42,
):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.int64)

    for tr, va in skf.split(X, y):
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=300,
            min_samples_leaf=10,
            l2_regularization=1e-3,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=15,
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])

    acc = accuracy_score(y, oof)
    mp = precision_score(y, oof, average="macro", zero_division=0)
    mf1 = f1_score(y, oof, average="macro", zero_division=0)
    rep = classification_report(y, oof, target_names=class_names, zero_division=0)

    final = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=10,
        l2_regularization=1e-3,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
    )
    final.fit(X, y)

    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(final, os.path.join(results_dir, "meta_gbm.pkl"))

    with open(os.path.join(results_dir, "meta_gbm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_accuracy": float(acc),
                "cv_macro_precision": float(mp),
                "cv_macro_f1": float(mf1),
            },
            f,
            indent=2,
        )

    with open(os.path.join(results_dir, "meta_gbm_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    print("[meta_gbm] CV acc:", acc, "macro-P:", mp, "macro-F1:", mf1)
    return final
