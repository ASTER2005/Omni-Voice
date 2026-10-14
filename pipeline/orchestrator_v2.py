"""
pipeline/orchestrator_v2.py
────────────────────────────────────────────────────────────────
Upgraded OmniVoice Pipeline with Streaming ASR + TTS

State machine:
  IDLE → VAD → PREPROCESSING → WAKE_WORD → SPEAKER_VERIFY
       → ACCEPTED → ASR_LISTEN → TTS_RESPOND → IDLE
       → REJECTED → TTS_DENY → IDLE

Features:
  • Always-on mic with Silero VAD
  • Wake word detection (ONNX)
  • Speaker verification (ECAPA-TDNN / ONNX)
  • Streaming ASR via faster-whisper (Whisper base)
  • TTS responses via edge-tts / pyttsx3
  • Pluggable on_command callback for custom logic

Usage:
    pipe = OmniVoicePipelineV2()
    pipe.run()
"""

import sys
import yaml
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from enum import Enum, auto
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_capture import AudioCapture
from preprocessing.vad import SileroVAD
from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor
from speaker_verification.verify import SpeakerVerifier
from asr.stream_asr import StreamingASR
from tts.speak import TTSSpeaker


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── State machine ────────────────────────────────────────────────────────────
class State(Enum):
    IDLE          = auto()
    VAD_ACTIVE    = auto()
    PREPROCESSING = auto()
    WAKE_WORD     = auto()
    VERIFICATION  = auto()
    ASR_LISTEN    = auto()
    TTS_RESPOND   = auto()
    ACCEPTED      = auto()
    REJECTED      = auto()


# ── Default TTS response templates ──────────────────────────────────────────
GREETINGS = [
    "Welcome back, {name}! I'm listening.",
    "Hello {name}, what can I do for you?",
    "Hi {name}! How can I help?",
]

REJECTION_MSGS = [
    "Sorry, I don't recognize your voice. Access denied.",
    "Voice not recognized. Please try again.",
    "Authentication failed. Unknown speaker.",
]


class OmniVoicePipelineV2:
    """
    Full OmniVoice pipeline with streaming ASR and TTS response.

    Args:
        cfg_path:           Path to config.yaml
        use_onnx_speaker:   Use ONNX model for speaker verification (faster)
        asr_model_size:     Whisper model size: 'tiny', 'base', 'small'
        asr_language:       Language code for ASR (e.g. 'en', 'hi')
        tts_voice:          edge-tts voice name (None = auto)
        tts_backend:        'edge' or 'pyttsx3' (None = auto-detect)
        on_command:         Callback(speaker_id: str, text: str) → str or None
                            Called with the transcribed command after verification.
                            Return a reply string to speak, or None to skip TTS.
    """

    def __init__(
        self,
        cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
        use_onnx_speaker: bool = False,
        asr_model_size: str = "base",
        asr_language: str = "en",
        tts_voice: Optional[str] = None,
        tts_backend: Optional[str] = None,
        on_command: Optional[Callable[[str, str], Optional[str]]] = None,
        skip_wake_word: bool = False,
    ):
        self.cfg_path = cfg_path
        self.skip_wake_word = skip_wake_word
        cfg = _load_cfg(cfg_path)
        self.ww_cfg = cfg["wake_word"]
        self.sp_cfg = cfg["speaker_verification"]
        self.asr_cfg = cfg.get("asr", {})
        self.tts_cfg = cfg.get("tts", {})
        log_cfg = cfg["logging"]

        # Logging
        Path(log_cfg["pipeline_log"]).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_cfg["level"], logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_cfg["pipeline_log"], encoding="utf-8"),
            ]
        )
        self.log = logging.getLogger("OmniVoiceV2")

        # ── Audio stack ──────────────────────────────────────────────────────
        self.capture  = AudioCapture(cfg_path)
        self.vad      = SileroVAD(cfg_path)
        self.nr       = NoiseReducer(cfg_path)
        self.fe       = FeatureExtractor(cfg_path)
        self.verifier = SpeakerVerifier(cfg_path, use_onnx=use_onnx_speaker)

        # ── Wake word ONNX session ───────────────────────────────────────────
        self._ww_session: Optional[ort.InferenceSession] = None
        self.ww_threshold = self.ww_cfg["threshold"]

        # ── ASR ──────────────────────────────────────────────────────────────
        _asr_size = self.asr_cfg.get("model_size", asr_model_size)
        _asr_lang = self.asr_cfg.get("language", asr_language)
        self.asr = StreamingASR(
            model_size=_asr_size,
            language=_asr_lang,
            device=self.asr_cfg.get("device", "cpu"),
            compute_type=self.asr_cfg.get("compute_type", "int8"),
            chunk_s=self.asr_cfg.get("chunk_s", 5.0),
        )

        # ── TTS ──────────────────────────────────────────────────────────────
        _tts_voice   = tts_voice   or self.tts_cfg.get("voice", None)
        _tts_backend = tts_backend or self.tts_cfg.get("backend", None)
        self.tts = TTSSpeaker(voice=_tts_voice, backend=_tts_backend)

        # ── Command handler ──────────────────────────────────────────────────
        self.on_command = on_command

        self.state = State.IDLE
        self._running = False
        self._greeting_idx = 0

    # ── Wake word ────────────────────────────────────────────────────────────
    def _load_ww_session(self) -> Optional[ort.InferenceSession]:
        onnx_path = self.ww_cfg["onnx_path"]
        if Path(onnx_path).exists():
            self.log.info(f"Loading wake word ONNX: {onnx_path}")
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.log.warning(
            f"Wake word ONNX not found at {onnx_path}. "
            "Run: python wake_word/export.py first."
        )
        return None

    def _detect_wake_word(self, audio: np.ndarray) -> tuple[bool, float]:
        if self._ww_session is None:
            return False, 0.0
        mel = self.fe.log_mel_wake_word(audio)
        inp = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
        logits = self._ww_session.run(None, {"mel_input": inp})[0]
        probs = self._softmax(logits[0])
        wake_prob = float(probs[1])
        return wake_prob >= self.ww_threshold, wake_prob

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Greeting helpers ─────────────────────────────────────────────────────
    def _greeting(self, name: str) -> str:
        msg = GREETINGS[self._greeting_idx % len(GREETINGS)].format(name=name)
        self._greeting_idx += 1
        return msg

    # ── Main pipeline loop ───────────────────────────────────────────────────
    def run(
        self,
        on_accept: Optional[Callable[[str, float], None]] = None,
        on_reject: Optional[Callable[[float], None]] = None,
        max_sessions: int = 0,    # 0 = run forever
    ) -> None:
        """
        Start the always-on OmniVoice pipeline.

        Full flow per session:
          1. Listen for wake word
          2. Verify speaker identity
          3. Greet accepted speaker via TTS
          4. Stream ASR until silence (one utterance)
          5. Call on_command(speaker_id, text) → TTS the reply
          6. Return to IDLE

        Args:
            on_accept:    Optional callback(speaker_id, score) on verification
            on_reject:    Optional callback(score) on rejection
            max_sessions: Stop after N complete sessions (0 = infinite)
        """
        self._ww_session = self._load_ww_session()
        self._running = True
        sessions = 0

        self.log.info("=" * 60)
        self.log.info("  OmniVoice V2  (ASR + TTS Enabled)")
        self.log.info(f"  Wake word:          {self.ww_cfg['target_words']}")
        self.log.info(f"  Enrolled speakers:  {list(self.verifier.enrolled.keys())}")
        self.log.info(f"  ASR model:          Whisper {self.asr._backend}")
        self.log.info(f"  TTS backend:        {self.tts._backend}")
        self.log.info("=" * 60)
        self.log.info("  Listening for wake word...")

        WINDOW = 16_000
        buffer = np.zeros(WINDOW, dtype=np.float32)
        chunk_size = self.capture.chunk_samples  # 480

        with self.capture:
            # ── Mic warm-up: flush device startup clicks / DC offset ───────────
            time.sleep(0.3)
            self.capture.flush()
            self.vad.reset_states()
            self.log.info("  Microphone ready. Speak anytime!")

            try:
                while self._running:
                    self.state = State.IDLE

                    # ── Step 1: VAD ──────────────────────────────────────────
                    chunk = self.capture.read()
                    is_sp, prob = self.vad.is_speech(chunk)
                    if not is_sp:
                        continue

                    self.state = State.VAD_ACTIVE

                    # ── Step 2: Rolling 1s buffer ────────────────────────────
                    self.state = State.PREPROCESSING
                    buffer = np.roll(buffer, -chunk_size)
                    buffer[-chunk_size:] = chunk
                    audio_1s = self.nr.process(buffer.copy())

                    # ── Step 3: Wake word / Direct speech onset ──────────────
                    if not self.skip_wake_word:
                        self.state = State.WAKE_WORD
                        triggered, ww_score = self._detect_wake_word(audio_1s)
                        if not triggered:
                            continue
                        self.log.info(f"🎙  Wake word! (p={ww_score:.3f})")

                    # ── Step 4: Collect complete utterance naturally ─────────
                    self.state = State.VERIFICATION
                    utterance_chunks = [chunk]
                    voiced_chunks_count = 1
                    silence_count = 0
                    max_silence = 18        # ~540ms silence to signal end of utterance
                    max_total_chunks = 250  # ~7.5 seconds max

                    while len(utterance_chunks) < max_total_chunks:
                        nxt_chunk = self.capture.read()
                        is_voiced, _ = self.vad.is_speech(nxt_chunk)
                        utterance_chunks.append(nxt_chunk)
                        if is_voiced:
                            voiced_chunks_count += 1
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= max_silence:
                                break

                    # Filter out isolated clicks, breaths, table taps:
                    # Genuine human speech requires at least 6 voiced chunks (~180ms)
                    # and total utterance length of at least 10 chunks (~300ms).
                    if voiced_chunks_count < 6 or len(utterance_chunks) < 10:
                        continue

                    if self.skip_wake_word:
                        self.log.info(f"🎙  Speech detected (voiced_chunks={voiced_chunks_count}, total={len(utterance_chunks)})")

                    audio_utterance = np.concatenate(utterance_chunks, axis=0)
                    self.vad.reset_states()

                    # Clean audio and extract voice features for verification
                    clean_audio = self.nr.process(audio_utterance)
                    voiced_audio = self.vad.extract_speech(clean_audio)
                    emb = self.verifier.embed(voiced_audio)
                    spk_id, spk_score = self.verifier.identify_best_match(emb)

                    if spk_id is None:
                        # ── REJECTED ─────────────────────────────────────────
                        self.state = State.REJECTED
                        self.log.warning(
                            f"✗ REJECTED | unknown speaker | score={spk_score:.4f}"
                        )
                        if on_reject:
                            on_reject(spk_score)
                        import random
                        deny_msg = random.choice(REJECTION_MSGS)
                        self.tts.say(deny_msg)
                        self.capture.flush()
                        self.vad.reset_states()
                        continue

                    # ── ACCEPTED ─────────────────────────────────────────────
                    self.state = State.ACCEPTED
                    self.log.info(
                        f"✓ ACCEPTED | speaker='{spk_id}' | score={spk_score:.4f}"
                    )
                    if on_accept:
                        on_accept(spk_id, spk_score)

                    # ── Step 5: ASR (Transcribe what you just said) ───────────
                    self.state = State.ASR_LISTEN
                    command_text = self.asr.transcribe_once(audio_utterance)

                    # ── Step 6: Generate reply & speak (TTS) ─────────────────
                    self.state = State.TTS_RESPOND
                    if command_text and len(command_text.strip()) > 1:
                        self.log.info(f"📝 Command: '{command_text}'")
                        if self.on_command:
                            reply = self.on_command(spk_id, command_text)
                        else:
                            reply = f"You said: {command_text}"
                    else:
                        # Speaker verified, but said only a quick wake-up word
                        reply = self._greeting(spk_id)

                    if reply:
                        self.log.info(f"🔊 Reply: {reply}")
                        self.tts.say(reply)

                    # Flush mic queue so AI doesn't listen to its own TTS output
                    self.capture.flush()
                    self.vad.reset_states()

                    sessions += 1
                    if max_sessions > 0 and sessions >= max_sessions:
                        break

            except KeyboardInterrupt:
                self.log.info("Pipeline stopped by user (Ctrl+C).")
            finally:
                self._running = False
                self.log.info("OmniVoice V2 Pipeline STOPPED.")

    def _listen_for_command(
        self,
        max_silence_chunks: int = 15,
        max_record_s: float = 10.0,
    ) -> Optional[str]:
        """
        Listen for a single command utterance after speaker is verified.

        Accumulates audio until silence is detected or max_record_s elapses.
        Returns the transcribed text or None if nothing was captured.
        """
        sr = 16_000
        max_samples = int(max_record_s * sr)
        buffer: list[np.ndarray] = []
        silence_count = 0
        recording = False
        total_samples = 0

        self.log.debug("_listen_for_command: waiting for speech...")

        while total_samples < max_samples:
            chunk = self.capture.read()
            is_sp, prob = self.vad.is_speech(chunk)
            total_samples += len(chunk)

            if is_sp:
                recording = True
                silence_count = 0
                buffer.append(chunk)
            else:
                if recording:
                    silence_count += 1
                    buffer.append(chunk)
                    if silence_count >= max_silence_chunks:
                        break  # End of utterance

        if not buffer:
            return None

        audio = np.concatenate(buffer, axis=0)
        try:
            text = self.asr.transcribe_once(audio)
            return text if text.strip() else None
        except Exception as e:
            self.log.error(f"ASR error: {e}")
            return None

    def stop(self) -> None:
        self._running = False


# ── Standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="OmniVoice V2 — ASR + TTS Pipeline")
    p.add_argument("--config",       default="C:/Omni_Voice/pipeline/config.yaml")
    p.add_argument("--onnx-speaker", action="store_true", help="Use ONNX speaker model")
    p.add_argument("--asr-model",    default="base",
                   choices=["tiny", "base", "small", "medium"],
                   help="Whisper model size")
    p.add_argument("--language",     default="en", help="ASR language code")
    p.add_argument("--tts-backend",  default=None,
                   choices=["edge", "pyttsx3"],
                   help="TTS backend (auto-detect if not set)")
    p.add_argument("--tts-voice",    default=None, help="TTS voice name")
    p.add_argument("--sessions",     type=int, default=0,
                   help="Number of sessions to run (0=infinite)")
    p.add_argument("--skip-wake-word", action="store_true",
                   help="Bypass wake-word detection and verify speaker immediately on speech")
    args = p.parse_args()

    # Example custom command handler
    def handle_command(speaker_id: str, text: str) -> Optional[str]:
        """Simple rule-based command handler. Replace with your LLM/NLP logic."""
        text_lower = text.lower()
        import datetime
        now = datetime.datetime.now()

        if any(w in text_lower for w in ["what time", "the time", "time now", "time is it"]):
            return f"The time is {now.strftime('%I:%M %p')}."
        elif any(w in text_lower for w in ["date", "today's date", "what day", "day is it"]):
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        elif any(w in text_lower for w in ["how are you", "how're you", "how do you do"]):
            return f"I'm doing great, {speaker_id}! How are you doing today?"
        elif any(w in text_lower for w in ["who are you", "your name", "what are you"]):
            return f"I am OmniVoice, your edge voice assistant. You are authenticated as {speaker_id}."
        elif any(w in text_lower for w in ["thank you", "thanks"]):
            return f"You're very welcome, {speaker_id}!"
        elif any(w in text_lower for w in ["hello", "hi", "hey"]):
            return f"Hello {speaker_id}! Nice to hear from you."
        elif any(w in text_lower for w in ["stop", "quit", "exit", "bye", "goodbye"]):
            return f"Goodbye {speaker_id}! Have a wonderful day."
        else:
            return f"I heard you say: {text}. Access granted for {speaker_id}."

    def on_accept(spk_id: str, score: float):
        print(f"\n  🟢 GRANTED → {spk_id}  (score={score:.4f})\n")

    def on_reject(score: float):
        print(f"\n  🔴 DENIED   (score={score:.4f})\n")

    pipe = OmniVoicePipelineV2(
        cfg_path=args.config,
        use_onnx_speaker=args.onnx_speaker,
        asr_model_size=args.asr_model,
        asr_language=args.language,
        tts_voice=args.tts_voice,
        tts_backend=args.tts_backend,
        on_command=handle_command,
        skip_wake_word=args.skip_wake_word,
    )
    pipe.run(
        on_accept=on_accept,
        on_reject=on_reject,
        max_sessions=args.sessions,
    )
