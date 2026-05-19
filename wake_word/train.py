"""
wake_word/train.py
────────────────────────────────────────────────────────────────
Training loop for the WakeWordCNN model.

Features:
  - AdamW + cosine annealing LR schedule
  - Cross-entropy with label smoothing
  - WeightedRandomSampler for class imbalance
  - Checkpointing (best val loss)
  - Tensorboard-compatible loss/accuracy logging
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_word.model import WakeWordCNN
from wake_word.dataset import WakeWordDataset


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> None:
    cfg = _load_cfg(cfg_path)
    t_cfg = cfg["train_wake_word"]
    ww_cfg = cfg["wake_word"]
    ckpt_dir = Path(t_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WakeWordTrain] Device: {device}")

    # ── Datasets & Loaders ─────────────────────────────────────────────────
    print("[WakeWordTrain] Building training dataset...")
    train_ds = WakeWordDataset(
        split=cfg["dataset"]["train_split"],
        cfg_path=cfg_path,
        augment=True,
    )
    val_ds = WakeWordDataset(
        split=cfg["dataset"]["dev_split"],
        cfg_path=cfg_path,
        augment=False,
    )

    sampler = train_ds.get_sampler()
    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    num_classes = len(ww_cfg["target_words"]) + 1   # wake_words + background
    model = WakeWordCNN(num_classes=num_classes).to(device)
    print(f"[WakeWordTrain] Parameters: {model.count_parameters():,}")

    # ── Loss, Optimizer, Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=t_cfg.get("label_smoothing", 0.1)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=1e-6
    )

    # ── Training Loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, t_cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{t_cfg['epochs']} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= total
        train_acc = correct / total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{t_cfg['epochs']} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "num_classes": num_classes,
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pth")
            # Also save to model_path from config
            model_path = Path(ww_cfg["model_path"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, model_path)
            print(f"  ✓ Best model saved → {model_path}")

    print(f"\n[WakeWordTrain] Training complete. "
          f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = parser.parse_args()
    train(cfg_path=args.config)
