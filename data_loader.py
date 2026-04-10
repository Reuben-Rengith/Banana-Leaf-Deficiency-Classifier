from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in VALID_EXTS


def count_per_class(dataset_dir: str, classes: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            out[cls] = 0
            continue
        out[cls] = sum(_is_image_file(f) for f in os.listdir(cls_dir))
    return out


def validate_dataset(dataset_dir: str, classes: List[str]) -> None:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset root not found: {dataset_dir}")
    missing = [c for c in classes if not os.path.isdir(os.path.join(dataset_dir, c))]
    if missing:
        raise FileNotFoundError(f"Missing class folders: {missing}")
    counts = count_per_class(dataset_dir, classes)
    print("[data] per-class counts:", counts)
    empty = [k for k, v in counts.items() if v == 0]
    if empty:
        raise ValueError(f"Empty class folders: {empty}")


def build_generators(
    dataset_dir: str,
    classes: List[str],
    img_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split,
        rotation_range=8,
        zoom_range=0.10,
        brightness_range=(0.90, 1.10),
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=val_split)

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        classes=classes,
        subset="training",
        shuffle=True,
        seed=seed,
        follow_links=True,
    )
    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        classes=classes,
        subset="validation",
        shuffle=False,
        seed=seed,
        follow_links=True,
    )

    if train_gen.samples == 0:
        raise ValueError("0 training images found.")
    if val_gen.samples == 0:
        raise ValueError("0 validation images found. Lower val split or verify data.")

    print("[data] class_indices:", train_gen.class_indices)
    print("[data] train samples:", train_gen.samples, "val samples:", val_gen.samples)
    return train_gen, val_gen


def class_weights_sparse(train_gen) -> Dict[int, float]:
    y = train_gen.classes
    cls = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=cls, y=y)
    return {int(c): float(v) for c, v in zip(cls, w)}


def focal_alpha_from_class_weights(num_classes: int, class_weights: Dict[int, float]) -> tf.Tensor:
    vec = np.ones((num_classes,), dtype=np.float32)
    for k, v in class_weights.items():
        vec[int(k)] = float(v)
    vec = vec / max(1e-8, float(vec.mean()))
    return tf.constant(vec, dtype=tf.float32)
