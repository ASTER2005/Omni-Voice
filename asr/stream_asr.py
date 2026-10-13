"""
asr/stream_asr.py
────────────────────────────────────────────────────────────────
Streaming / chunk-based ASR using OpenAI Whisper (tiny or base).

Strategy:
  • After speaker is ACCEPTED by the pipeline, this module
    captures audio in rolling chunks and transcribes them
    in near-real-time using Whisper's encoder-decoder.
  • Uses faster-whisper (CTranslate2 backend) for ~4× speedup
    over the original PyTorch Whisper on CPU.
  • Falls back to openai-whisper if faster-whisper is not installed.

Usage (standalone):
    asr = StreamingASR()
    text = asr.transcribe_once(audio_np_16khz)   # one shot
    # OR stream from mic:
    for partial in asr.stream(capture, vad):
        print(partial)
"""

import sys
import numpy as np
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Backend selection ────────────────────────────────────────────────────────
def _load_faster_whisper(model_size: str, device: str, compute_type: str):
    """Try faster-whisper first (CTranslate2 — much faster on CPU)."""
    from faster_whisper import WhisperModel
    print(f"[ASR] Loading faster-whisper '{model_size}' on {device} ({compute_type})...")
    return WhisperModel(model_size, device=device, compute_type=compute_type), "faster"


def _load_openai_whisper(model_size: str):
    """Fallback: original OpenAI Whisper."""
    import whisper
    print(f"[ASR] Loading openai-whisper '{model_size}'...")
    model = whisper.load_model(model_size)
    return model, "openai"


class StreamingASR:
    """
    Near-real-time ASR engine wrapping Whisper.

    Args:
        model_size:    Whisper model size: 'tiny', 'base', 'small' (default: 'base')
        language:      Language code ('en', 'hi', etc.) or None for auto-detect
        device:        'cpu' or 'cuda'
        compute_type:  CTranslate2 quantisation: 'int8', 'float16', 'float32'
        chunk_s:       Duration of each audio chunk fed to Whisper (seconds)
        min_silence_s: Silence gap that ends a utterance window (seconds)
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        chunk_s: float = 5.0,
        min_silence_s: float = 0.8,
    ):
        self.language = language
        self.chunk_samples = int(chunk_s * self.SAMPLE_RATE)
        self.min_silence_samples = int(min_silence_s * self.SAMPLE_RATE)
        self._backend = None
        self._model = None

        # Try faster-whisper first, fall back to openai-whisper
        try:
            self._model, self._backend = _load_faster_whisper(
                model_size, device, compute_type
            )
        except ImportError:
            print("[ASR] faster-whisper not found. Falling back to openai-whisper.")
            try:
                self._model, self._backend = _load_openai_whisper(model_size)
            except ImportError:
                raise ImportError(
                    "[ASR] Neither faster-whisper nor openai-whisper is installed.\n"
                    "  Install:  pip install faster-whisper\n"
                    "        OR  pip install openai-whisper"
                )

        print(f"[ASR] Ready — backend={self._backend}, language={language}")

    # ── Single-shot transcription ────────────────────────────────────────────
    def transcribe_once(self, audio: np.ndarray) -> str:
        """
        Transcribe a single audio array (float32, 16kHz) to text.

        Args:
            audio: 1-D float32 numpy array at 16kHz sample rate

        Returns:
            Transcribed text string (stripped).
        """
        audio = audio.astype(np.float32)
        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        if self._backend == "faster":
            segments, _ = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=3,
                vad_filter=True,          # built-in silero VAD
                vad_parameters=dict(
                    min_silence_duration_ms=400,
                    speech_pad_ms=200,
                ),
            )
            return " ".join(s.text.strip() for s in segments).strip()

        else:  # openai backend
            result = self._model.transcribe(
                audio,
                language=self.language,
                fp16=False,
            )
            return result["text"].strip()

    # ── Streaming from mic (generator) ──────────────────────────────────────
    def stream(
        self,
        capture,          # AudioCapture instance
        vad,              # SileroVAD instance
        max_silence_chunks: int = 15,   # ~450ms silence → flush
    ) -> Iterator[str]:
        """
        Yield partial transcription strings as speech is detected.

        Continuously reads from the mic, accumulates audio while
        speech is active, and transcribes each utterance when
        silence is detected.

        Args:
            capture:             AudioCapture instance (already opened)
            vad:                 SileroVAD instance
            max_silence_chunks:  Number of silent chunks before flushing buffer

        Yields:
            Transcribed text for each detected utterance.
        """
        buffer: list[np.ndarray] = []
        silence_count = 0
        recording = False

        print("[ASR] Streaming started — speak anytime (Ctrl+C to stop)...")

        while True:
            chunk = capture.read()
            is_speech, prob = vad.is_speech(chunk)

            if is_speech:
                if not recording:
                    print(f"[ASR] 🎤 Speech detected (p={prob:.2f})...")
                    recording = True
                buffer.append(chunk)
                silence_count = 0
            else:
                if recording:
                    silence_count += 1
                    buffer.append(chunk)  # include trailing silence
                    if silence_count >= max_silence_chunks:
                        # Flush and transcribe
                        audio = np.concatenate(buffer, axis=0)
                        text = self.transcribe_once(audio)
                        if text:
                            yield text
                        buffer = []
                        silence_count = 0
                        recording = False
                        vad.reset_states()

    # ── Transcribe from a pre-recorded file ─────────────────────────────────
    def transcribe_file(self, wav_path: str) -> str:
        """Transcribe an audio file (.wav, .mp3, etc.)."""
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        if sr != self.SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
            except ImportError:
                raise RuntimeError(
                    f"Audio is {sr}Hz but Whisper needs 16kHz. "
                    "Install librosa to auto-resample."
                )
        return self.transcribe_once(audio)


# ── Standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, "C:/Omni_Voice")

    from preprocessing.audio_capture import AudioCapture
    from preprocessing.vad import SileroVAD

    p = argparse.ArgumentParser(description="OmniVoice Streaming ASR demo")
    p.add_argument("--model",    default="base",  help="Whisper model size (tiny/base/small)")
    p.add_argument("--language", default="en",    help="Language code (en, hi, ...)")
    p.add_argument("--file",     default=None,    help="Transcribe a wav file instead of mic")
    p.add_argument("--config",   default="C:/Omni_Voice/pipeline/config.yaml")
    args = p.parse_args()

    asr = StreamingASR(model_size=args.model, language=args.language)

    if args.file:
        print(f"\n📝 Transcription: {asr.transcribe_file(args.file)}\n")
    else:
        cap = AudioCapture(args.config)
        vad = SileroVAD(args.config)
        with cap:
            try:
                for text in asr.stream(cap, vad):
                    print(f"\n📝 You said: {text}\n")
            except KeyboardInterrupt:
                print("\n[ASR] Stopped.")
