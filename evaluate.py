from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras import Model

from config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE_EFF,
    IMG_SIZE_MOB,
    RESULTS_DIR,
    SEED,
    VAL_SPLIT,
    active_classes,
)
from data_loader import build_generators
from feature_fusion import build_meta_features, extract_gap_batch, predict_probs_batch
from visualizations import plot_confusion, plot_roc_multiclass, plot_tsne, save_report


def load_encoder(classifier):
    return Model(classifier.input, classifier.get_layer("gap").output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    classes = active_classes()
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _, val_m = build_generators(
        args.data_dir, classes, IMG_SIZE_MOB, BATCH_SIZE, VAL_SPLIT, SEED
    )
    _, val_e = build_generators(
        args.data_dir, classes, IMG_SIZE_EFF, BATCH_SIZE, VAL_SPLIT, SEED
    )

    if list(val_m.filepaths) != list(val_e.filepaths):
        raise ValueError("Validation file ordering mismatch between generators.")

    model_m = tf.keras.models.load_model(
        os.path.join(args.results_dir, "mobilenet_unified_final.h5"),
        compile=False,
    )
    model_e = tf.keras.models.load_model(
        os.path.join(args.results_dir, "efficientnet_unified_final.h5"),
        compile=False,
    )
    gbm = joblib.load(os.path.join(args.results_dir, "meta_gbm.pkl"))
    enc_m = load_encoder(model_m)
    enc_e = load_encoder(model_e)

    paths = val_m.filepaths
    y = np.asarray(val_m.classes)
    pm = predict_probs_batch(model_m, paths, IMG_SIZE_MOB)
    pe = predict_probs_batch(model_e, paths, IMG_SIZE_EFF)
    gm = extract_gap_batch(enc_m, paths, IMG_SIZE_MOB)
    ge = extract_gap_batch(enc_e, paths, IMG_SIZE_EFF)
    X = build_meta_features(gm, ge, pm, pe)

    pred = gbm.predict(X)
    prob = gbm.predict_proba(X)

    print(classification_report(y, pred, target_names=classes, zero_division=0))
    print("acc:", accuracy_score(y, pred), "macro-F1:", f1_score(y, pred, average="macro", zero_division=0))

    save_report(y, pred, classes, "gbm_eval", plots_dir)
    plot_confusion(y, pred, classes, "gbm_eval", plots_dir, normalize=False)
    plot_confusion(y, pred, classes, "gbm_eval", plots_dir, normalize=True)
    plot_roc_multiclass(y, prob, classes, "gbm_eval", plots_dir)
    plot_tsne(X, y, classes, "gbm_eval", plots_dir)
    print("Saved plots/reports to:", plots_dir)


if __name__ == "__main__":
    main()
