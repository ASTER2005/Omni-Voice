"""
evaluation/plots.py
Publication-quality ROC, DET curves, confusion matrix, latency bar chart.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path


sns.set_theme(style="darkgrid", palette="muted", font_scale=1.2)
RESULTS_DIR = Path("C:/Omni_Voice/evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc(fpr, tpr, auc_score, title="ROC Curve", save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0",
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0,1],[0,1],"--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{title.replace(' ','_').lower()}.png"
        fig.savefig(str(path), dpi=150)
        print(f"[Plot] Saved → {path}")
    return fig


def plot_det(fpr, fnr, title="DET Curve", save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr * 100, fnr * 100, lw=2, color="#DD8452")
    ax.set_xlabel("False Accept Rate (%)")
    ax.set_ylabel("False Reject Rate (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{title.replace(' ','_').lower()}.png"
        fig.savefig(str(path), dpi=150)
        print(f"[Plot] Saved → {path}")
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save=True):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{title.replace(' ','_').lower()}.png"
        fig.savefig(str(path), dpi=150)
        print(f"[Plot] Saved → {path}")
    return fig


def plot_latency(stage_names, latency_ms, title="Per-Stage Latency", save=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", len(stage_names))
    bars = ax.barh(stage_names, latency_ms, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f ms", padding=4)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(title)
    ax.axvline(x=10, color="red", linestyle="--", lw=1.5, label="10ms target")
    ax.legend()
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{title.replace(' ','_').lower()}.png"
        fig.savefig(str(path), dpi=150)
        print(f"[Plot] Saved → {path}")
    return fig


def plot_training_history(history: dict, title="Training History", save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()

    if "val_acc" in history:
        ax2.plot(epochs, history.get("train_acc", [0]*len(epochs)), label="Train Acc")
        ax2.plot(epochs, history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy"); ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{title.replace(' ','_').lower()}.png"
        fig.savefig(str(path), dpi=150)
        print(f"[Plot] Saved → {path}")
    return fig
