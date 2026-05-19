"""
speaker_verification/enroll.py
────────────────────────────────────────────────────────────────
LIVE microphone enrollment:
  1. Prompt user to speak K utterances
  2. Capture each utterance via sounddevice + silero-vad
  3. Extract ECAPA/ResNet-18 embedding per utterance
  4. Average embeddings → save as speaker_id.npy
"""

import sys
import time
import yaml
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_capture import AudioCapture
from preprocessing.vad import SileroVAD
from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor
from speaker_verification.model import ResNet18SpeakerEncoder


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_encoder(cfg: dict, device: torch.device) -> ResNet18SpeakerEncoder:
    """Load the trained speaker encoder from checkpoint."""
    model_path = cfg["speaker_verification"]["model_path"]
    emb_dim = cfg["speaker_verification"]["embedding_dim"]

    encoder = ResNet18SpeakerEncoder(embedding_dim=emb_dim).to(device)

    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        print(f"[Enroll] Loaded encoder from {model_path}")
    else:
        print(f"[Enroll] ⚠  No trained model at {model_path}. "
              f"Using random weights (train the model first).")

    encoder.eval()
    return encoder


def capture_utterance(
    capture: AudioCapture,
    vad: SileroVAD,
    nr: NoiseReducer,
    duration_s: float,
    utterance_idx: int,
) -> np.ndarray:
    """
    Capture one voiced utterance from the live microphone.
    Waits for speech onset, then collects `duration_s` of audio.
    """
    print(f"\n  [Utterance {utterance_idx}] Speak now... (listening for voice)")
    capture.flush()

    # Wait for speech onset
    while True:
        chunk = capture.read()
        is_sp, prob = vad.is_speech(chunk)
        if is_sp:
            print(f"  ✓ Voice detected (p={prob:.2f}). Recording {duration_s}s...")
            break

    # Capture full utterance
    audio = capture.read_seconds(duration_s)
    audio = nr.process(audio)
    vad.reset_states()
    return audio


def embed_audio(
    audio: np.ndarray,
    encoder: ResNet18SpeakerEncoder,
    fe: FeatureExtractor,
    device: torch.device,
) -> np.ndarray:
    """Convert raw audio to L2-normalised embedding."""
    mel = fe.mel_speaker(audio)                         # [80, T]
    mel = fe.FeatureExtractor._pad_or_clip if False else _pad_mel(mel, 300)
    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,80,T]
    with torch.no_grad():
        emb = encoder(tensor)                           # [1, D]
    return emb.squeeze(0).cpu().numpy()                 # [D]


def _pad_mel(mel: np.ndarray, max_frames: int) -> np.ndarray:
    if mel.shape[1] >= max_frames:
        return mel[:, :max_frames]
    return np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])))


def enroll(
    speaker_id: str,
    cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
) -> str:
    """
    Full live enrollment flow.

    Returns path to saved embedding file.
    """
    cfg = _load_cfg(cfg_path)
    sp_cfg = cfg["speaker_verification"]
    enrolled_dir = Path(sp_cfg["enrolled_dir"])
    enrolled_dir.mkdir(parents=True, exist_ok=True)

    K = sp_cfg["enroll_utterances"]           # default 7
    dur = sp_cfg["enroll_duration_s"]         # default 4.0s

    device = torch.device("cpu")
    encoder = _load_encoder(cfg, device)
    fe = FeatureExtractor(cfg_path)
    nr = NoiseReducer(cfg_path)
    vad = SileroVAD(cfg_path)

    print(f"\n{'='*55}")
    print(f"  SPEAKER ENROLLMENT — ID: {speaker_id!r}")
    print(f"  You will be asked to speak {K} utterances of ~{dur}s each.")
    print(f"  Please speak clearly after each prompt.")
    print(f"{'='*55}")
    time.sleep(1)

    embeddings: List[np.ndarray] = []

    with AudioCapture(cfg_path) as capture:
        for i in range(1, K + 1):
            audio = capture_utterance(capture, vad, nr, dur, i)
            emb = embed_audio(audio, encoder, fe, device)
            embeddings.append(emb)
            print(f"  ✓ Utterance {i}/{K} encoded | emb norm={np.linalg.norm(emb):.4f}")
            time.sleep(0.5)

    # Average all K embeddings
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    # L2-normalise the mean
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)

    save_path = enrolled_dir / f"{speaker_id}.npy"
    np.save(str(save_path), mean_emb)

    print(f"\n  ✓ Speaker '{speaker_id}' enrolled → {save_path}")
    print(f"    Embedding dim: {mean_emb.shape[0]}")
    print(f"    Averaged over: {K} utterances\n")

    return str(save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",     required=True, help="Speaker identifier")
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = parser.parse_args()
    enroll(speaker_id=args.id, cfg_path=args.config)
