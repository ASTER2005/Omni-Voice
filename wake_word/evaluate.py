"""
wake_word/evaluate.py
────────────────────────────────────────────────────────────────
Evaluate the trained wake word model on test-clean split.
Reports: FAR, FRR, EER, accuracy, confusion matrix.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_word.model import WakeWordCNN
from wake_word.dataset import WakeWordDataset


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(
    cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
    model_path: str | None = None,
) -> dict:
    cfg = _load_cfg(cfg_path)
    model_path = model_path or cfg["wake_word"]["model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location=device)
    num_classes = ckpt.get("num_classes", 2)
    model = WakeWordCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[WakeWordEval] Loaded model from {model_path}")

    # ── Test dataset ───────────────────────────────────────────────────────
    test_ds = WakeWordDataset(
        split=cfg["dataset"]["test_split"],
        cfg_path=cfg_path,
        augment=False,
    )
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # ── Inference ──────────────────────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            probs = model.predict_proba(x)              # [B, num_classes]
            preds = probs.argmax(dim=1).cpu().numpy()
            wake_prob = probs[:, 1].cpu().numpy()       # probability of wake word class
            all_labels.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(wake_prob)

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    # ── Metrics ────────────────────────────────────────────────────────────
    # FAR = FP / (FP + TN)  — imposter accepted rate
    # FRR = FN / (FN + TP)  — genuine rejected rate
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (fp + tn + 1e-9)
    frr = fn / (fn + tp + 1e-9)
    acc = (tp + tn) / len(labels)

    # EER from ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    roc_auc = auc(fpr, tpr)

    results = {
        "accuracy":  acc,
        "FAR":       far,
        "FRR":       frr,
        "EER":       eer,
        "AUC":       roc_auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    print("\n" + "═" * 50)
    print("  WAKE WORD EVALUATION RESULTS")
    print("═" * 50)
    print(f"  Accuracy :  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  FAR      :  {far:.4f}  ({far*100:.2f}%)  [target ≤ 1%]")
    print(f"  FRR      :  {frr:.4f}  ({frr*100:.2f}%)  [target ≤ 5%]")
    print(f"  EER      :  {eer:.4f}  ({eer*100:.2f}%)")
    print(f"  AUC-ROC  :  {roc_auc:.4f}")
    print("═" * 50)
    print(classification_report(labels, preds, target_names=["Background", "Wake Word"]))

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    parser.add_argument("--model",  default=None)
    args = parser.parse_args()
    evaluate(cfg_path=args.config, model_path=args.model)
