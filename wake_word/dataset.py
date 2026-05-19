"""
wake_word/dataset.py
────────────────────────────────────────────────────────────────
LibriSpeech-based wake word dataset.

Strategy:
  POSITIVE: Scan .trans.txt files for target word occurrences,
            use torchaudio forced aligner (CTC) to locate word
            timestamps, extract 1-second windows.
  NEGATIVE: Random 1-second windows from non-keyword utterances.

Ratio: 10 negatives per positive (configurable via config.yaml).
"""

import os
import re
import random
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Forced Alignment Helpers ────────────────────────────────────────────────
def _find_word_in_transcript(text: str, target: str) -> List[int]:
    """Return list of word-level positions (0-based) where target appears."""
    words = text.upper().split()
    target = target.upper()
    return [i for i, w in enumerate(words) if w == target]


def _extract_window_around_word(
    audio: np.ndarray,
    sr: int,
    word_idx: int,
    words: List[str],
    duration_s: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Heuristic word timing: assume ~300ms/word (average English speech rate).
    Extract `duration_s` window centred on the word.
    """
    ms_per_word = 300  # ms
    centre_ms = word_idx * ms_per_word + ms_per_word // 2
    half = int(sr * duration_s / 2)
    centre_sample = int(centre_ms * sr / 1000)
    start = max(0, centre_sample - half)
    end = start + int(sr * duration_s)
    if end > len(audio):
        return None
    return audio[start:end]


# ── Dataset Builder ─────────────────────────────────────────────────────────
class WakeWordDatasetBuilder:
    """
    Scans LibriSpeech split and builds lists of
    (audio_path, start_sample, end_sample, label) tuples.
    Saves/loads a cache to avoid re-scanning.
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)
        self.ds_root = Path(cfg["dataset"]["root"])
        self.processed_dir = Path(cfg["dataset"]["processed_dir"])
        self.target_words: List[str] = [
            w.upper() for w in cfg["wake_word"]["target_words"]
        ]
        self.sr: int = cfg["audio"]["sample_rate"]
        self.duration_s: float = cfg["wake_word"]["input_duration_s"]
        self.neg_ratio: int = cfg["train_wake_word"]["negative_ratio"]
        (self.processed_dir / "wake_word").mkdir(parents=True, exist_ok=True)

    def _split_dir(self, split: str) -> Path:
        return self.ds_root / split / "LibriSpeech" / split

    def scan_split(self, split: str, cache: bool = True) -> Tuple[List, List]:
        """
        Returns (positive_items, negative_items).
        Each item: dict with keys 'path', 'start', 'end'.
        """
        cache_file = self.processed_dir / "wake_word" / f"{split}_index.pkl"
        if cache and cache_file.exists():
            print(f"[WakeWordDataset] Loading cached index: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        positives, negatives = [], []
        split_dir = self._split_dir(split)

        all_trans = list(split_dir.rglob("*.trans.txt"))
        print(f"[WakeWordDataset] Scanning {len(all_trans)} transcript files "
              f"in {split} for words: {self.target_words}")

        for trans_path in tqdm(all_trans, desc=f"Scanning {split}"):
            audio_dir = trans_path.parent
            with open(trans_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts[0], parts[1]
                flac_path = audio_dir / f"{utt_id}.flac"
                if not flac_path.exists():
                    continue

                words = text.upper().split()
                target_len = int(self.sr * self.duration_s)

                # Check for target words
                found = False
                for tw in self.target_words:
                    positions = _find_word_in_transcript(text, tw)
                    for pos in positions:
                        audio, _ = sf.read(str(flac_path), dtype="float32")
                        if audio.ndim > 1:
                            audio = audio[:, 0]
                        win = _extract_window_around_word(
                            audio, self.sr, pos, words, self.duration_s
                        )
                        if win is not None:
                            positives.append({
                                "path": str(flac_path),
                                "audio": win,
                                "label": 1,
                            })
                            found = True

                if not found:
                    # Use as negative: random 1s window
                    try:
                        info = sf.info(str(flac_path))
                        total = info.frames
                        if total >= target_len:
                            start = random.randint(0, total - target_len)
                            negatives.append({
                                "path": str(flac_path),
                                "start": start,
                                "end": start + target_len,
                                "label": 0,
                            })
                    except Exception:
                        pass

        print(f"[WakeWordDataset] {split}: {len(positives)} positives, "
              f"{len(negatives)} negatives")

        result = (positives, negatives)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result


# ── PyTorch Dataset ─────────────────────────────────────────────────────────
class WakeWordDataset(Dataset):
    """
    PyTorch Dataset for wake word detection.
    Returns (log_mel_tensor [1,64,101], label_int).
    """

    def __init__(
        self,
        split: str = "train-clean-360",
        cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
        augment: bool = True,
    ):
        cfg = _load_cfg(cfg_path)
        self.sr: int = cfg["audio"]["sample_rate"]
        self.duration_s: float = cfg["wake_word"]["input_duration_s"]
        self.neg_ratio: int = cfg["train_wake_word"]["negative_ratio"]
        self.augment = augment
        self.aug_cfg = cfg["train_wake_word"]["augment"]

        builder = WakeWordDatasetBuilder(cfg_path)
        positives, negatives = builder.scan_split(split)

        # Balance: keep neg_ratio negatives per positive
        n_neg = min(len(negatives), len(positives) * self.neg_ratio)
        random.shuffle(negatives)
        negatives = negatives[:n_neg]

        self.items = positives + negatives
        random.shuffle(self.items)

        self.fe = FeatureExtractor(cfg_path)
        self.nr = NoiseReducer(cfg_path)

        # Weights for WeightedRandomSampler
        label_counts = {0: n_neg, 1: len(positives)}
        self.sample_weights = [
            1.0 / label_counts[item["label"]] for item in self.items
        ]
        print(f"[WakeWordDataset] {split}: {len(positives)} pos + "
              f"{n_neg} neg = {len(self.items)} total")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        label = item["label"]

        # Load audio
        if "audio" in item:
            audio = item["audio"].copy()
        else:
            audio, _ = sf.read(
                item["path"],
                start=item["start"],
                stop=item["end"],
                dtype="float32",
            )
            if audio.ndim > 1:
                audio = audio[:, 0]

        # Noise reduction
        audio = self.nr.process(audio)

        # Augmentation
        if self.augment:
            audio = self._augment(audio)

        # Feature extraction → [64, 101]
        mel = self.fe.log_mel_wake_word(audio)

        # SpecAugment
        if self.augment and self.aug_cfg.get("specaugment", False):
            mel = self.fe.spec_augment(
                mel,
                time_mask_max=self.aug_cfg.get("time_mask_max", 30),
                freq_mask_max=self.aug_cfg.get("freq_mask_max", 20),
            )

        # → [1, 64, 101] (add channel dim)
        tensor = torch.from_numpy(mel).unsqueeze(0)
        return tensor, label

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Speed perturbation only (noise mixing handled offline)."""
        cfg = self.aug_cfg
        if cfg.get("speed_perturb", False):
            lo, hi = cfg.get("speed_range", [0.9, 1.1])
            rate = random.uniform(lo, hi)
            if abs(rate - 1.0) > 0.02:
                audio = FeatureExtractor.speed_perturb(audio, self.sr, rate)
        return audio

    def get_sampler(self) -> WeightedRandomSampler:
        weights = torch.tensor(self.sample_weights, dtype=torch.float)
        return WeightedRandomSampler(weights, num_samples=len(self.items), replacement=True)
