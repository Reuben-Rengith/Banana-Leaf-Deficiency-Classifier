from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from config import (
    IMG_SIZE_EFF,
    IMG_SIZE_MOB,
    RESULTS_DIR,
    active_classes,
)
from feature_fusion import build_meta_features, extract_gap_batch, predict_probs_batch


def load_encoder(classifier):
    return Model(classifier.input, classifier.get_layer("gap").output)


def predict_image(image_path: str, results_dir: str):
    classes = active_classes()
    model_m = tf.keras.models.load_model(
        os.path.join(results_dir, "mobilenet_unified_final.h5"), compile=False
    )
    model_e = tf.keras.models.load_model(
        os.path.join(results_dir, "efficientnet_unified_final.h5"), compile=False
    )
    gbm = joblib.load(os.path.join(results_dir, "meta_gbm.pkl"))
    enc_m = load_encoder(model_m)
    enc_e = load_encoder(model_e)

    pm = predict_probs_batch(model_m, [image_path], IMG_SIZE_MOB)
    pe = predict_probs_batch(model_e, [image_path], IMG_SIZE_EFF)
    gm = extract_gap_batch(enc_m, [image_path], IMG_SIZE_MOB)
    ge = extract_gap_batch(enc_e, [image_path], IMG_SIZE_EFF)
    X = build_meta_features(gm, ge, pm, pe)

    proba = gbm.predict_proba(X)[0]
    idx = np.argsort(proba)[::-1][:3]
    return [(classes[i], float(proba[i])) for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    top3 = predict_image(args.image, args.results_dir)
    print("Top-3 predictions:", top3)


if __name__ == "__main__":
    main()
