from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def _mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_history(history, name: str, out_dir: str):
    _mkdir(out_dir)
    h = history.history
    if "accuracy" in h and "val_accuracy" in h:
        plt.figure(figsize=(7, 5))
        plt.plot(h["accuracy"], label="train_acc")
        plt.plot(h["val_accuracy"], label="val_acc")
        plt.title(f"{name} Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_accuracy.png"), dpi=220)
        plt.close()
    if "loss" in h and "val_loss" in h:
        plt.figure(figsize=(7, 5))
        plt.plot(h["loss"], label="train_loss")
        plt.plot(h["val_loss"], label="val_loss")
        plt.title(f"{name} Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_loss.png"), dpi=220)
        plt.close()


def save_report(y_true, y_pred, class_names, name: str, out_dir: str):
    _mkdir(out_dir)
    txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(out_dir, f"{name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def plot_confusion(y_true, y_pred, class_names, name: str, out_dir: str, normalize: bool = False):
    _mkdir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{name} Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.tight_layout()
    suffix = "_norm" if normalize else ""
    plt.savefig(os.path.join(out_dir, f"{name}_cm{suffix}.png"), dpi=220)
    plt.close()


def plot_roc_multiclass(y_true, y_prob, class_names, name: str, out_dir: str):
    _mkdir(out_dir)
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n))
    plt.figure(figsize=(9, 7))
    for i in range(n):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{name} ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_roc.png"), dpi=220)
    plt.close()


def plot_tsne(X, y_true, class_names, name: str, out_dir: str, max_points: int = 2500):
    _mkdir(out_dir)
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X = X[idx]
        y_true = y_true[idx]
    Z = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
    plt.figure(figsize=(9, 7))
    for i, cls in enumerate(class_names):
        m = y_true == i
        if np.any(m):
            plt.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.7, label=cls)
    plt.legend(fontsize=8)
    plt.title(f"{name} t-SNE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_tsne.png"), dpi=220)
    plt.close()
