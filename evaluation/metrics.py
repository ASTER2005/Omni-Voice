"""
evaluation/metrics.py — FAR, FRR, EER, min-DCF calculations.
evaluation/plots.py    — ROC, DET curves and latency plots.
"""

# ════════════════════════════════════════════════════════════════
#  evaluation/metrics.py
# ════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Tuple


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate.
    Returns (eer, threshold_at_eer).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute minimum Detection Cost Function (min-DCF)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return float(dcf.min())


def compute_far_frr(
    labels: np.ndarray,
    preds: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute FAR and FRR from binary predictions.
    FAR = FP / (FP + TN) — false accept rate
    FRR = FN / (FN + TP) — false reject rate
    """
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    far = fp / (fp + tn + 1e-9)
    frr = fn / (fn + tp + 1e-9)
    return float(far), float(frr)


def compute_roc(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, thresholds, auc_score)."""
    fpr, tpr, th = roc_curve(labels, scores)
    return fpr, tpr, th, auc(fpr, tpr)
