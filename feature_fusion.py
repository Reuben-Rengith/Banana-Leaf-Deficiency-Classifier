from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing import image as keras_image


def entropy_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1, keepdims=True)


def max_confidence(p: np.ndarray) -> np.ndarray:
    return np.max(p, axis=1, keepdims=True)


def predict_probs_batch(model, paths: list[str], img_size: tuple[int, int]) -> np.ndarray:
    out = []
    for p in paths:
        im = keras_image.load_img(p, target_size=img_size)
        x = keras_image.img_to_array(im) / 255.0
        out.append(model.predict(np.expand_dims(x, 0), verbose=0)[0])
    return np.array(out)


def extract_gap_batch(encoder, paths: list[str], img_size: tuple[int, int]) -> np.ndarray:
    out = []
    for p in paths:
        im = keras_image.load_img(p, target_size=img_size)
        x = keras_image.img_to_array(im) / 255.0
        out.append(encoder.predict(np.expand_dims(x, 0), verbose=0)[0])
    return np.array(out)


def build_meta_features(
    gap_mob: np.ndarray,
    gap_eff: np.ndarray,
    prob_mob: np.ndarray,
    prob_eff: np.ndarray,
) -> np.ndarray:
    h_m = entropy_probs(prob_mob)
    h_e = entropy_probs(prob_eff)
    mx_m = max_confidence(prob_mob)
    mx_e = max_confidence(prob_eff)
    mn_m = np.mean(prob_mob, axis=1, keepdims=True)
    mn_e = np.mean(prob_eff, axis=1, keepdims=True)
    return np.concatenate([gap_mob, gap_eff, h_m, h_e, mx_m, mx_e, mn_m, mn_e], axis=1)
