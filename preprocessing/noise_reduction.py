"""
preprocessing/noise_reduction.py
────────────────────────────────────────────────────────────────
Noise-robust audio preprocessing:
  1. Bandpass filter  (scipy Butterworth 300–3400 Hz)
  2. Spectral gating  (noisereduce)
  3. Amplitude normalisation
"""

import numpy as np
import scipy.signal as signal
import noisereduce as nr
import yaml


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NoiseReducer:
    """
    Two-stage noise reduction pipeline:
      Stage 1 — Butterworth bandpass filter
      Stage 2 — Spectral gating via noisereduce

    Usage:
        nr = NoiseReducer()
        clean = nr.process(noisy_audio_float32)
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)["audio"]
        self.sr: int = cfg["sample_rate"]
        self.low_hz: float = float(cfg["bandpass_low_hz"])
        self.high_hz: float = float(cfg["bandpass_high_hz"])
        self._sos = self._design_bandpass()
        print(f"[NoiseReducer] Bandpass: {self.low_hz}–{self.high_hz} Hz "
              f"@ {self.sr} Hz SR")

    # ── Private helpers ─────────────────────────────────────────────────────
    def _design_bandpass(self) -> np.ndarray:
        """Design a 4th-order Butterworth bandpass filter (SOS form)."""
        nyq = self.sr / 2.0
        low = self.low_hz / nyq
        high = self.high_hz / nyq
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        return sos

    # ── Public API ──────────────────────────────────────────────────────────
    def bandpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to audio."""
        return signal.sosfiltfilt(self._sos, audio).astype(np.float32)

    def spectral_gate(
        self,
        audio: np.ndarray,
        stationary: bool = True,
        prop_decrease: float = 1.0,
    ) -> np.ndarray:
        """
        Apply noisereduce spectral gating.

        Args:
            audio:          float32 waveform
            stationary:     True for stationary noise (faster)
            prop_decrease:  how much to reduce noise (0–1, default 1.0 = full)
        """
        reduced = nr.reduce_noise(
            y=audio,
            sr=self.sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
        )
        return reduced.astype(np.float32)

    def normalize(self, audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
        """RMS-normalize audio to a target level."""
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        return (audio * (target_rms / rms)).astype(np.float32)

    def process(self, audio: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Full pipeline: bandpass → spectral gate → normalize.

        Args:
            audio:     float32 mono waveform at self.sr
            normalize: apply RMS normalization at the end

        Returns:
            Cleaned float32 waveform.
        """
        audio = self.bandpass(audio)
        audio = self.spectral_gate(audio)
        if normalize:
            audio = self.normalize(audio)
        return audio

    def add_noise_snr(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """
        Mix clean + noise at a given SNR (dB).
        Used for data augmentation.

        Args:
            clean:   clean waveform float32
            noise:   noise waveform float32 (will be looped/trimmed)
            snr_db:  target Signal-to-Noise Ratio in dB

        Returns:
            Noisy mixture float32.
        """
        # Align lengths
        if len(noise) < len(clean):
            reps = int(np.ceil(len(clean) / len(noise)))
            noise = np.tile(noise, reps)
        noise = noise[: len(clean)]

        clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-9
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
        snr_linear = 10 ** (snr_db / 20.0)
        noise_scaled = noise * (clean_rms / (noise_rms * snr_linear))
        return (clean + noise_scaled).astype(np.float32)


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import soundfile as sf
    import sys

    reducer = NoiseReducer()
    if len(sys.argv) > 1:
        audio, sr = sf.read(sys.argv[1])
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        clean = reducer.process(audio)
        out_path = sys.argv[1].replace(".flac", "_clean.wav").replace(".wav", "_clean.wav")
        sf.write(out_path, clean, sr)
        print(f"Saved cleaned audio → {out_path}")
    else:
        dummy = np.random.randn(16000).astype(np.float32) * 0.1
        clean = reducer.process(dummy)
        print(f"Noise reduction demo | in RMS: {dummy.std():.4f} | "
              f"out RMS: {clean.std():.4f}")
