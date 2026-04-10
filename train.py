from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from config import (
    BATCH_SIZE,
    CE_MIX,
    DATA_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS_FT1,
    EPOCHS_FT2,
    EPOCHS_HEAD,
    FOCAL_GAMMA,
    IMG_SIZE_EFF,
    IMG_SIZE_MOB,
    RESULTS_DIR,
    SEED,
    VAL_SPLIT,
    active_classes,
)
from data_loader import (
    build_generators,
    class_weights_sparse,
    focal_alpha_from_class_weights,
    validate_dataset,
)
from feature_fusion import build_meta_features, extract_gap_batch, predict_probs_batch
from meta_gbm import train_meta_gbm
from models.backbones import (
    build_efficientnet_unified,
    build_mobilenet_unified,
    sparse_focal_plus_ce,
    unfreeze_last_n,
)
from visualizations import plot_confusion, plot_history, plot_roc_multiclass, plot_tsne, save_report


def build_callbacks(results_dir: str, name: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, f"{name}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    classes = active_classes()
    num_classes = len(classes)

    validate_dataset(args.data_dir, classes)

    train_m, val_m = build_generators(
        args.data_dir, classes, IMG_SIZE_MOB, BATCH_SIZE, VAL_SPLIT, SEED
    )
    train_e, val_e = build_generators(
        args.data_dir, classes, IMG_SIZE_EFF, BATCH_SIZE, VAL_SPLIT, SEED
    )

    cw_m = class_weights_sparse(train_m)
    cw_e = class_weights_sparse(train_e)
    alpha_e = focal_alpha_from_class_weights(num_classes, cw_e)

    # MobileNet training
    model_m, enc_m, base_m = build_mobilenet_unified(
        num_classes=num_classes, img_size=IMG_SIZE_MOB, lr=1e-4
    )
    h_m1 = model_m.fit(
        train_m,
        validation_data=val_m,
        epochs=EPOCHS_HEAD,
        class_weight=cw_m,
        callbacks=build_callbacks(args.results_dir, "mob_p1"),
        verbose=1,
    )
    plot_history(h_m1, "mobilenet_p1", plots_dir)

    unfreeze_last_n(base_m, n=60)
    model_m.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h_m2 = model_m.fit(
        train_m,
        validation_data=val_m,
        epochs=EPOCHS_HEAD + EPOCHS_FT1,
        initial_epoch=EPOCHS_HEAD,
        class_weight=cw_m,
        callbacks=build_callbacks(args.results_dir, "mob_p2"),
        verbose=1,
    )
    plot_history(h_m2, "mobilenet_p2", plots_dir)
    model_m.save(os.path.join(args.results_dir, "mobilenet_unified_final.h5"))

    # EfficientNet training
    model_e, enc_e, base_e = build_efficientnet_unified(
        num_classes=num_classes,
        img_size=IMG_SIZE_EFF,
        lr=3e-4,
        gamma=FOCAL_GAMMA,
        alpha_per_class=alpha_e,
        ce_mix=CE_MIX,
    )
    h_e1 = model_e.fit(
        train_e,
        validation_data=val_e,
        epochs=EPOCHS_HEAD,
        class_weight=cw_e,
        callbacks=build_callbacks(args.results_dir, "eff_p1"),
        verbose=1,
    )
    plot_history(h_e1, "efficientnet_p1", plots_dir)

    unfreeze_last_n(base_e, n=60)
    model_e.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=sparse_focal_plus_ce(
            gamma=FOCAL_GAMMA,
            alpha_per_class=alpha_e,
            ce_weight=CE_MIX,
        ),
        metrics=["accuracy"],
    )
    h_e2 = model_e.fit(
        train_e,
        validation_data=val_e,
        epochs=EPOCHS_HEAD + EPOCHS_FT1,
        initial_epoch=EPOCHS_HEAD,
        class_weight=cw_e,
        callbacks=build_callbacks(args.results_dir, "eff_p2"),
        verbose=1,
    )
    plot_history(h_e2, "efficientnet_p2", plots_dir)

    unfreeze_last_n(base_e, n=80)
    model_e.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-4),
        loss=sparse_focal_plus_ce(
            gamma=FOCAL_GAMMA,
            alpha_per_class=alpha_e,
            ce_weight=CE_MIX,
        ),
        metrics=["accuracy"],
    )
    h_e3 = model_e.fit(
        train_e,
        validation_data=val_e,
        epochs=EPOCHS_HEAD + EPOCHS_FT1 + EPOCHS_FT2,
        initial_epoch=EPOCHS_HEAD + EPOCHS_FT1,
        class_weight=cw_e,
        callbacks=build_callbacks(args.results_dir, "eff_p3"),
        verbose=1,
    )
    plot_history(h_e3, "efficientnet_p3", plots_dir)
    model_e.save(os.path.join(args.results_dir, "efficientnet_unified_final.h5"))

    # Meta learner on fused train features
    train_paths = train_m.filepaths
    y_train = np.asarray(train_m.classes)
    gm = extract_gap_batch(enc_m, train_paths, IMG_SIZE_MOB)
    ge = extract_gap_batch(enc_e, train_paths, IMG_SIZE_EFF)
    pm = predict_probs_batch(model_m, train_paths, IMG_SIZE_MOB)
    pe = predict_probs_batch(model_e, train_paths, IMG_SIZE_EFF)
    X_train = build_meta_features(gm, ge, pm, pe)
    gbm = train_meta_gbm(X_train, y_train, classes, args.results_dir, cv_splits=5, seed=SEED)

    # Validation evaluation and plots
    val_paths = val_m.filepaths
    y_val = np.asarray(val_m.classes)
    vm = predict_probs_batch(model_m, val_paths, IMG_SIZE_MOB)
    ve = predict_probs_batch(model_e, val_paths, IMG_SIZE_EFF)
    gmv = extract_gap_batch(enc_m, val_paths, IMG_SIZE_MOB)
    gev = extract_gap_batch(enc_e, val_paths, IMG_SIZE_EFF)
    X_val = build_meta_features(gmv, gev, vm, ve)
    pred = gbm.predict(X_val)
    prob = gbm.predict_proba(X_val)

    print("[val] acc:", accuracy_score(y_val, pred), "macro-F1:", f1_score(y_val, pred, average="macro", zero_division=0))

    save_report(y_val, pred, classes, "gbm_val", plots_dir)
    plot_confusion(y_val, pred, classes, "gbm_val", plots_dir, normalize=False)
    plot_confusion(y_val, pred, classes, "gbm_val", plots_dir, normalize=True)
    plot_roc_multiclass(y_val, prob, classes, "gbm_val", plots_dir)
    plot_tsne(X_val, y_val, classes, "gbm_val", plots_dir)

    print("Training complete. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
