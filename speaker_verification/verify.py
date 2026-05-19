"""
speaker_verification/verify.py
────────────────────────────────────────────────────────────────
LIVE microphone speaker verification:
  1. Capture one utterance from mic (VAD-gated)
  2. Extract embedding with trained encoder
  3. Cosine similarity against enrolled speaker embeddings
  4. Accept if similarity ≥ threshold, reject otherwise
"""

import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_capture import AudioCapture
from preprocessing.vad import SileroVAD
from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor
from speaker_verification.model import ResNet18SpeakerEncoder


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pad_mel(mel: np.ndarray, max_frames: int = 300) -> np.ndarray:
    if mel.shape[1] >= max_frames:
        return mel[:, :max_frames]
    return np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])))


class SpeakerVerifier:
    """
    Runtime speaker verification engine.

    Loads all enrolled speaker embeddings into memory.
    Verifies live mic audio against enrolled speakers.
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        self.cfg_path = cfg_path
        cfg = _load_cfg(cfg_path)
        self.sp_cfg = cfg["speaker_verification"]
        self.threshold: float = self.sp_cfg["threshold"]
        self.enrolled_dir = Path(self.sp_cfg["enrolled_dir"])
        self.device = torch.device("cpu")

        self.fe = FeatureExtractor(cfg_path)
        self.nr = NoiseReducer(cfg_path)
        self.vad = SileroVAD(cfg_path)

        # Load encoder
        self.encoder = self._load_encoder(cfg)

        # Load all enrolled embeddings
        self.enrolled: Dict[str, np.ndarray] = {}
        self.reload_enrolled()

    def _load_encoder(self, cfg: dict) -> ResNet18SpeakerEncoder:
        emb_dim = self.sp_cfg["embedding_dim"]
        encoder = ResNet18SpeakerEncoder(embedding_dim=emb_dim).to(self.device)
        model_path = Path(self.sp_cfg["model_path"])
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=self.device)
            encoder.load_state_dict(ckpt["encoder_state"])
            print(f"[Verifier] Encoder loaded from {model_path}")
        else:
            print("[Verifier] ⚠  No trained encoder found. Using random weights.")
        encoder.eval()
        return encoder

    def reload_enrolled(self) -> None:
        """Reload enrolled speaker embeddings from disk."""
        self.enrolled.clear()
        if not self.enrolled_dir.exists():
            print("[Verifier] No enrolled speakers directory found.")
            return
        for npy in self.enrolled_dir.glob("*.npy"):
            spk_id = npy.stem
            self.enrolled[spk_id] = np.load(str(npy))
        print(f"[Verifier] Loaded {len(self.enrolled)} enrolled speakers: "
              f"{list(self.enrolled.keys())}")

    # ── Core embedding ──────────────────────────────────────────────────────
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Convert raw audio to L2-normalised speaker embedding."""
        mel = self.fe.mel_speaker(audio)
        mel = _pad_mel(mel, 300)
        t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder(t).squeeze(0).cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    # ── Cosine similarity ───────────────────────────────────────────────────
    def cosine_similarity(
        self, emb: np.ndarray, speaker_id: str
    ) -> float:
        """Cosine similarity between query embedding and enrolled speaker."""
        enrolled_emb = self.enrolled.get(speaker_id)
        if enrolled_emb is None:
            raise ValueError(f"Speaker '{speaker_id}' not enrolled.")
        return float(np.dot(emb, enrolled_emb))

    def identify_best_match(
        self, emb: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching enrolled speaker and their similarity score.
        Returns (speaker_id, score) or (None, score) if below threshold.
        """
        if not self.enrolled:
            return None, 0.0
        best_id, best_score = None, -1.0
        for spk_id, spk_emb in self.enrolled.items():
            score = float(np.dot(emb, spk_emb))
            if score > best_score:
                best_score = score
                best_id = spk_id
        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score

    # ── Live verification (single known speaker) ────────────────────────────
    def verify_live(
        self,
        speaker_id: str,
        capture: AudioCapture,
        duration_s: float = 4.0,
    ) -> Tuple[bool, float]:
        """
        Verify live mic audio against a specific enrolled speaker.

        Returns:
            (accepted: bool, similarity: float)
        """
        print(f"\n[Verify] Verifying speaker '{speaker_id}' — speak now...")
        capture.flush()

        while True:
            chunk = capture.read()
            is_sp, _ = self.vad.is_speech(chunk)
            if is_sp:
                break

        audio = capture.read_seconds(duration_s)
        audio = self.nr.process(audio)
        self.vad.reset_states()

        emb = self.embed(audio)
        score = self.cosine_similarity(emb, speaker_id)
        accepted = score >= self.threshold

        status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
        print(f"[Verify] {status} | speaker='{speaker_id}' | "
              f"score={score:.4f} | threshold={self.threshold:.4f}")
        return accepted, score

    # ── Live identification (open-set) ──────────────────────────────────────
    def identify_live(
        self,
        capture: AudioCapture,
        duration_s: float = 4.0,
    ) -> Tuple[Optional[str], float]:
        """
        Identify live mic speaker from all enrolled speakers.

        Returns:
            (speaker_id or None, best_score)
        """
        print("\n[Identify] Listening for speaker...")
        capture.flush()

        while True:
            chunk = capture.read()
            is_sp, _ = self.vad.is_speech(chunk)
            if is_sp:
                break

        audio = capture.read_seconds(duration_s)
        audio = self.nr.process(audio)
        self.vad.reset_states()

        emb = self.embed(audio)
        best_id, best_score = self.identify_best_match(emb)

        if best_id:
            print(f"[Identify] ✓ Identified: '{best_id}' | score={best_score:.4f}")
        else:
            print(f"[Identify] ✗ Unknown speaker | best_score={best_score:.4f}")

        return best_id, best_score


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",     required=True, help="Enrolled speaker ID to verify")
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = parser.parse_args()

    verifier = SpeakerVerifier(args.config)
    with AudioCapture(args.config) as cap:
        accepted, score = verifier.verify_live(args.id, cap)
