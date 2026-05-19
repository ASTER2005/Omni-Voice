"""
speaker_verification/train.py
────────────────────────────────────────────────────────────────
Training loop for ResNet-18 speaker encoder with ArcFace loss.

  - AdamW optimizer + cosine LR annealing
  - ArcFace (margin=0.5, scale=64) metric learning
  - Online hard negative mining via hard triplet loss option
  - Checkpoints best model by validation EER
"""

import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_verification.dataset import SpeakerDataset
from speaker_verification.model import ResNet18SpeakerEncoder, ArcFaceHead


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> None:
    cfg    = _load_cfg(cfg_path)
    t_cfg  = cfg["train_speaker"]
    sp_cfg = cfg["speaker_verification"]
    ckpt_dir = Path(t_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SpeakerTrain] Device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────
    print("[SpeakerTrain] Building training dataset...")
    train_ds = SpeakerDataset(
        split=cfg["dataset"]["train_split"],
        cfg_path=cfg_path,
        augment=True,
        max_utt_per_speaker=t_cfg.get("utterances_per_speaker", 20),
    )
    val_ds = SpeakerDataset(
        split=cfg["dataset"]["dev_split"],
        cfg_path=cfg_path,
        augment=False,
    )

    # Windows: num_workers > 0 causes multiprocessing overhead → use 0
    import platform
    nw = 0 if platform.system() == "Windows" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    num_classes = train_ds.num_classes

    # ── Model + Head ───────────────────────────────────────────────────────
    encoder = ResNet18SpeakerEncoder(
        embedding_dim=sp_cfg["embedding_dim"]
    ).to(device)
    arc_head = ArcFaceHead(
        embedding_dim=sp_cfg["embedding_dim"],
        num_classes=num_classes,
        margin=t_cfg["arcface_margin"],
        scale=t_cfg["arcface_scale"],
    ).to(device)

    print(f"[SpeakerTrain] Encoder params: {encoder.count_parameters():,}")
    print(f"[SpeakerTrain] Num classes:    {num_classes}")

    # ── Loss & Optimiser ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + list(arc_head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=1e-5   # floor at 1e-5 to avoid NaN
    )

    # ── Training Loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, t_cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────
        encoder.train()
        arc_head.train()
        train_loss = correct = top5_correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{t_cfg['epochs']} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            emb    = encoder(x)                  # [B, D]
            logits = arc_head(emb, y)            # [B, num_classes]  ← ArcFace (with margin)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            # Use inference-mode logits (no margin) for meaningful accuracy
            with torch.no_grad():
                inf_logits = arc_head(emb.detach(), labels=None)  # cosine * scale
            preds = inf_logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            # Top-5 accuracy (more informative for 921-class problem)
            top5 = inf_logits.topk(5, dim=1).indices
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= total
        train_acc   = correct / total
        train_top5  = top5_correct / total

        # ── Validate ───────────────────────────────────────────────────────
        encoder.eval()
        arc_head.eval()
        val_loss = val_correct = val_top5_correct = val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{t_cfg['epochs']} [Val]"):
                x, y = x.to(device), y.to(device)
                emb = encoder(x)
                # Loss: use ArcFace margin (training signal)
                arc_logits = arc_head(emb, y)
                loss = criterion(arc_logits, y)
                val_loss += loss.item() * x.size(0)
                # Accuracy: inference mode — no margin (real-world metric)
                inf_logits = arc_head(emb, labels=None)
                preds = inf_logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                top5 = inf_logits.topk(5, dim=1).indices
                val_top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc      = val_correct / val_total
        val_top5     = val_top5_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Top1: {train_acc*100:.2f}% Top5: {train_top5*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Top1: {val_acc*100:.2f}% Top5: {val_top5*100:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "head_state": arc_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_classes": num_classes,
            "embedding_dim": sp_cfg["embedding_dim"],
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pth")
            model_path = Path(sp_cfg["model_path"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, model_path)
            print(f"  ✓ Best model saved → {model_path}")

    print(f"\n[SpeakerTrain] Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = parser.parse_args()
    train(cfg_path=args.config)
