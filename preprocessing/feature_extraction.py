"""
preprocessing/feature_extraction.py
────────────────────────────────────────────────────────────────
Audio feature extraction:
  - Log-Mel spectrogram  (wake word:          64 bands, 1s → [64×101])
  - Mel filterbank       (speaker verif.:     80 bands, variable-length)
  - MFCC                 (13/40 coefficients)
  - CQCC                 (Constant-Q Cepstral Coefficients)
"""

import numpy as np
import librosa
import yaml


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FeatureExtractor:
    """
    Unified feature extractor for wake word and speaker verification.

    Usage:
        fe = FeatureExtractor()
        mel_ww   = fe.log_mel_wake_word(audio)    # → [64, 101] float32
        mel_spk  = fe.mel_speaker(audio)           # → [80,  T] float32
        mfcc     = fe.mfcc(audio)                  # → [40,  T] float32
        cqcc     = fe.cqcc(audio)                  # → [28,  T] float32
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)
        aw = cfg["wake_word"]
        asp = cfg["speaker_verification"]
        self.sr: int = cfg["audio"]["sample_rate"]

        # Wake word params
        self.ww_n_mels: int = aw["n_mels"]          # 64
        self.ww_hop: int = aw["hop_length"]          # 160 = 10ms
        self.ww_win: int = aw["win_length"]          # 400 = 25ms
        self.ww_nfft: int = aw["n_fft"]             # 512
        self.ww_dur: float = aw["input_duration_s"] # 1.0s → 101 frames

        # Speaker verification params
        self.sp_n_mels: int = asp["n_mels"]          # 80
        self.sp_hop: int = asp["hop_length"]          # 160
        self.sp_win: int = asp["win_length"]          # 400
        self.sp_nfft: int = asp["n_fft"]             # 512

    # ── Wake Word Feature ──────────────────────────────────────────────────
    def log_mel_wake_word(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 64-band log-Mel spectrogram for wake word detection.

        Args:
            audio: float32 waveform, expected ~1s (16000 samples)

        Returns:
            float32 array of shape [n_mels, time_frames] ≈ [64, 101]
        """
        target_len = int(self.sr * self.ww_dur)
        audio = self._pad_or_trim(audio, target_len)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.ww_nfft,
            hop_length=self.ww_hop,
            win_length=self.ww_win,
            n_mels=self.ww_n_mels,
            fmin=80.0,
            fmax=7600.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)

    # ── Speaker Verification Feature ───────────────────────────────────────
    def mel_speaker(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 80-band Mel filterbank for speaker verification.

        Args:
            audio: float32 waveform (variable length)

        Returns:
            float32 array of shape [n_mels, time_frames] ≈ [80, T]
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.sp_nfft,
            hop_length=self.sp_hop,
            win_length=self.sp_win,
            n_mels=self.sp_n_mels,
            fmin=80.0,
            fmax=7600.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)

    # ── MFCC ──────────────────────────────────────────────────────────────
    def mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC coefficients (+ delta + delta-delta if n_mfcc ≤ 13).

        Returns:
            float32 array of shape [n_mfcc, T]
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.sp_nfft,
            hop_length=self.sp_hop,
            win_length=self.sp_win,
        )
        return mfcc.astype(np.float32)

    # ── CQCC ──────────────────────────────────────────────────────────────
    def cqcc(self, audio: np.ndarray, n_bins: int = 84, n_cqcc: int = 28) -> np.ndarray:
        """
        Constant-Q Cepstral Coefficients.

        Returns:
            float32 array of shape [n_cqcc, T]
        """
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.sp_hop,
            n_bins=n_bins,
        ))
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        # DCT to get cepstral coefficients
        from scipy.fftpack import dct
        cqcc_feats = dct(log_cqt, type=2, axis=0, norm="ortho")[:n_cqcc]
        return cqcc_feats.astype(np.float32)

    # ── Specaugment ────────────────────────────────────────────────────────
    def spec_augment(
        self,
        spec: np.ndarray,
        time_mask_max: int = 30,
        freq_mask_max: int = 20,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
    ) -> np.ndarray:
        """
        SpecAugment: random time and frequency masking.

        Args:
            spec: spectrogram [freq_bins, time_frames]
        """
        spec = spec.copy()
        n_freq, n_time = spec.shape

        for _ in range(n_time_masks):
            t = np.random.randint(0, min(time_mask_max, n_time))
            t0 = np.random.randint(0, max(1, n_time - t))
            spec[:, t0:t0 + t] = spec.min()

        for _ in range(n_freq_masks):
            f = np.random.randint(0, min(freq_mask_max, n_freq))
            f0 = np.random.randint(0, max(1, n_freq - f))
            spec[f0:f0 + f, :] = spec.min()

        return spec

    # ── Utility ────────────────────────────────────────────────────────────
    @staticmethod
    def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
        """Pad with zeros or trim to exact length."""
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len]

    @staticmethod
    def speed_perturb(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        """
        Speed perturbation via librosa time-stretch + resample.
        rate < 1.0 → slower, rate > 1.0 → faster.
        """
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        resampled = librosa.resample(stretched, orig_sr=int(sr * rate), target_sr=sr)
        return resampled.astype(np.float32)


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import soundfile as sf
    import sys

    fe = FeatureExtractor()
    if len(sys.argv) > 1:
        audio, sr = sf.read(sys.argv[1])
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        ww = fe.log_mel_wake_word(audio)
        sp = fe.mel_speaker(audio)
        mf = fe.mfcc(audio)
        print(f"Wake-word Mel: {ww.shape}  |  Speaker Mel: {sp.shape}  |  MFCC: {mf.shape}")
    else:
        dummy = np.random.randn(16000).astype(np.float32) * 0.1
        ww = fe.log_mel_wake_word(dummy)
        sp = fe.mel_speaker(dummy)
        print(f"Wake-word Mel: {ww.shape}  |  Speaker Mel: {sp.shape}")
