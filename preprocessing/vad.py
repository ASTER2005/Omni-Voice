"""
preprocessing/vad.py
────────────────────────────────────────────────────────────────
Voice Activity Detection using Silero-VAD.
Filters audio chunks — only passes speech frames to downstream.
"""

import numpy as np
import torch
import yaml
from typing import List, Tuple


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SileroVAD:
    """
    Wrapper around Silero-VAD for real-time and offline speech detection.

    Usage (real-time, chunk-by-chunk):
        vad = SileroVAD()
        prob = vad.is_speech(chunk_float32_16khz)   # → True/False

    Usage (offline, full waveform):
        timestamps = vad.get_speech_timestamps(waveform, sample_rate=16000)
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)["vad"]
        self.threshold: float = cfg["threshold"]
        self.min_speech_ms: int = cfg["min_speech_duration_ms"]
        self.min_silence_ms: int = cfg["min_silence_duration_ms"]
        self.window_size: int = cfg["window_size_samples"]   # 512 for 16kHz
        self.sample_rate: int = 16000

        print("[VAD] Loading Silero-VAD model...")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._vad_iterator,
            self._collect_chunks,
        ) = self._utils
        self._model.eval()
        self._iterator = self._vad_iterator(self._model, sampling_rate=self.sample_rate)
        print("[VAD] Silero-VAD ready.")

    # ── Real-time: one chunk at a time ──────────────────────────────────────
    def is_speech(self, chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Classify a single audio chunk.

        Args:
            chunk: float32 array of shape (N,) at 16kHz

        Returns:
            (is_speech: bool, probability: float)
        """
        tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, N]
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return prob >= self.threshold, prob

    def reset_states(self) -> None:
        """Reset VAD internal states (call between utterances)."""
        self._model.reset_states()

    # ── Offline: full waveform → speech segments ─────────────────────────
    def get_speech_timestamps(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        return_seconds: bool = False,
    ) -> List[dict]:
        """
        Run VAD on a full waveform and return list of speech segments.

        Returns:
            List of {'start': int, 'end': int} sample indices (or seconds).
        """
        tensor = torch.from_numpy(waveform).float()
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=return_seconds,
        )
        return timestamps

    def extract_speech(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract and concatenate all voiced segments from a waveform.
        """
        timestamps = self.get_speech_timestamps(waveform, sample_rate)
        if not timestamps:
            return waveform   # fallback: return original if no speech found
        tensor = torch.from_numpy(waveform).float()
        speech = self._collect_chunks(timestamps, tensor)
        return speech.numpy()


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import soundfile as sf
    import sys

    vad = SileroVAD()
    if len(sys.argv) > 1:
        audio, sr = sf.read(sys.argv[1])
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        speech = vad.extract_speech(audio, sr)
        print(f"Original: {len(audio)/sr:.2f}s | Speech: {len(speech)/sr:.2f}s")
    else:
        # Simulate random audio
        dummy = np.random.randn(16000).astype(np.float32) * 0.01
        is_sp, prob = vad.is_speech(dummy)
        print(f"Silence test → is_speech={is_sp}, prob={prob:.4f}")
