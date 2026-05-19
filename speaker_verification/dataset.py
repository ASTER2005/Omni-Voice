"""
speaker_verification/dataset.py
────────────────────────────────────────────────────────────────
LibriSpeech multi-speaker dataset for speaker verification training.

  - train-clean-360 : 921 speaker classes → ArcFace training
  - dev-clean        : 40  speakers       → threshold tuning
  - test-clean       : 40  speakers       → imposter evaluation

Returns speaker-labelled Mel filterbank spectrograms.
"""

import os
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Speaker Index Builder ────────────────────────────────────────────────────
class LibriSpeechIndex:
    """
    Scans a LibriSpeech split and builds:
      speaker_id (int 0..N-1) → list of flac file paths
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)
        self.root = Path(cfg["dataset"]["root"])
        self.processed_dir = Path(cfg["dataset"]["processed_dir"])
        (self.processed_dir / "speaker_verification").mkdir(parents=True, exist_ok=True)

    def _split_dir(self, split: str) -> Path:
        return self.root / split / "LibriSpeech" / split

    def build(self, split: str, cache: bool = True) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
        """
        Returns:
            idx2files : {speaker_idx: [flac_path, ...]}
            spk2idx   : {speaker_folder_name: speaker_idx}
        """
        cache_file = self.processed_dir / "speaker_verification" / f"{split}_index.pkl"
        if cache and cache_file.exists():
            print(f"[SpeakerIndex] Loading cached index: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        split_dir = self._split_dir(split)
        speaker_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"[SpeakerIndex] Found {len(speaker_dirs)} speakers in {split}")

        idx2files: Dict[int, List[str]] = {}
        spk2idx:   Dict[str, int] = {}

        for idx, spk_dir in enumerate(tqdm(speaker_dirs, desc=f"Indexing {split}")):
            spk_name = spk_dir.name
            spk2idx[spk_name] = idx
            flac_files = list(spk_dir.rglob("*.flac"))
            idx2files[idx] = [str(f) for f in flac_files]

        result = (idx2files, spk2idx)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result


# ── PyTorch Dataset ─────────────────────────────────────────────────────────
class SpeakerDataset(Dataset):
    """
    Returns (mel_tensor [1, 80, T], speaker_label_int) pairs.

    For variable-length audio, T is clipped/padded to max_frames.
    """

    def __init__(
        self,
        split: str = "train-clean-360",
        cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
        augment: bool = True,
        max_frames: int = 300,          # ~3s at 10ms hop
        max_utt_per_speaker: int = 20,
    ):
        cfg = _load_cfg(cfg_path)
        self.sr:         int   = cfg["audio"]["sample_rate"]
        self.augment:    bool  = augment
        self.max_frames: int   = max_frames
        self.noise_snr:  list  = cfg["train_speaker"].get(
            "noise_snr_range", [5, 20]
        )

        self.fe = FeatureExtractor(cfg_path)
        self.nr = NoiseReducer(cfg_path)

        indexer = LibriSpeechIndex(cfg_path)
        idx2files, _ = indexer.build(split)
        self.num_classes = len(idx2files)

        # Build flat list: (path, speaker_idx)
        self.items: List[Tuple[str, int]] = []
        for spk_idx, files in idx2files.items():
            random.shuffle(files)
            for f in files[:max_utt_per_speaker]:
                self.items.append((f, spk_idx))

        random.shuffle(self.items)

        # Collect all flac paths for cross-speaker noise mixing
        self._all_paths = [p for p, _ in self.items]

        print(f"[SpeakerDataset] {split}: {self.num_classes} speakers, "
              f"{len(self.items)} utterances")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, spk_label = self.items[idx]

        # Load audio
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Noise reduction + normalize
        audio = self.nr.process(audio)

        # Augmentation: cross-speaker noise mixing
        if self.augment:
            audio = self._add_cross_speaker_noise(audio)

        # Extract 80-band Mel filterbank
        mel = self.fe.mel_speaker(audio)              # [80, T]

        # Clip or pad to max_frames
        mel = self._pad_or_clip(mel, self.max_frames)

        # SpecAugment
        if self.augment:
            mel = self.fe.spec_augment(mel, time_mask_max=30, freq_mask_max=10)

        tensor = torch.from_numpy(mel).unsqueeze(0)   # [1, 80, max_frames]
        return tensor, spk_label

    def _pad_or_clip(self, mel: np.ndarray, max_frames: int) -> np.ndarray:
        n_frames = mel.shape[1]
        if n_frames >= max_frames:
            start = random.randint(0, n_frames - max_frames)
            return mel[:, start:start + max_frames]
        else:
            pad = max_frames - n_frames
            return np.pad(mel, ((0, 0), (0, pad)), mode="constant")

    def _add_cross_speaker_noise(self, clean: np.ndarray) -> np.ndarray:
        """Add a random LibriSpeech utterance as background noise."""
        try:
            noise_path = random.choice(self._all_paths)
            noise, _ = sf.read(noise_path, dtype="float32")
            if noise.ndim > 1:
                noise = noise[:, 0]
            snr = random.uniform(*self.noise_snr)
            return self.nr.add_noise_snr(clean, noise, snr)
        except Exception:
            return clean
