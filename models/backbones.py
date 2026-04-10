from __future__ import annotations

import keras.backend as K
import tensorflow as tf
from tensorflow.keras import Model, layers


def sparse_categorical_focal_loss(gamma: float = 2.0, alpha_per_class: tf.Tensor | None = None):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        if alpha_per_class is None:
            alpha_t = tf.ones_like(p_t, dtype=tf.float32)
        else:
            alpha_t = tf.gather(alpha_per_class, y_true)
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))

    return loss


def sparse_focal_plus_ce(
    gamma: float = 2.0,
    alpha_per_class: tf.Tensor | None = None,
    ce_weight: float = 0.30,
):
    focal_fn = sparse_categorical_focal_loss(gamma=gamma, alpha_per_class=alpha_per_class)

    def loss(y_true, y_pred):
        y_true_i = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_i, y_pred))
        focal = focal_fn(y_true, y_pred)
        return (1.0 - ce_weight) * focal + ce_weight * ce

    return loss


def unfreeze_last_n(base_model, n: int = 60):
    for layer in base_model.layers[:-n]:
        layer.trainable = False
    for layer in base_model.layers[-n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


def build_mobilenet_unified(num_classes: int, img_size=(224, 224), lr: float = 1e-4):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    base.trainable = False
    inp = layers.Input(shape=(*img_size, 3))
    x = base(inp, training=False)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    h = layers.Dense(256, activation="relu")(gap)
    h = layers.Dropout(0.35)(h)
    out = layers.Dense(num_classes, activation="softmax")(h)
    classifier = Model(inp, out, name="mobilenet_unified")
    encoder = Model(inp, gap, name="mobilenet_encoder")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return classifier, encoder, base


def build_efficientnet_unified(
    num_classes: int,
    img_size=(224, 224),
    lr: float = 3e-4,
    gamma: float = 2.0,
    alpha_per_class: tf.Tensor | None = None,
    ce_mix: float = 0.30,
):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    base.trainable = False
    inp = layers.Input(shape=(*img_size, 3))
    x = base(inp, training=False)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    h = layers.BatchNormalization()(gap)
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.45)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.30)(h)
    out = layers.Dense(num_classes, activation="softmax")(h)
    classifier = Model(inp, out, name="efficientnet_unified")
    encoder = Model(inp, gap, name="efficientnet_encoder")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=sparse_focal_plus_ce(gamma=gamma, alpha_per_class=alpha_per_class, ce_weight=ce_mix),
        metrics=["accuracy"],
    )
    return classifier, encoder, base
