"""
pipeline/orchestrator.py
────────────────────────────────────────────────────────────────
End-to-end state machine:
  IDLE → VAD → PREPROCESSING → WAKE_WORD → SPEAKER_VERIFY → ACCEPT/REJECT

Always-on microphone stream. ONNX models for fast CPU inference.
"""

import sys, yaml, time, logging, queue
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


def _load_cfg(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class State(Enum):
    IDLE          = auto()
    VAD_ACTIVE    = auto()
    PREPROCESSING = auto()
    WAKE_WORD     = auto()
    VERIFICATION  = auto()
    ACCEPTED      = auto()
    REJECTED      = auto()


class OmniVoicePipeline:
    """
    Full edge voice processing pipeline.

    Usage:
        pipe = OmniVoicePipeline()
        pipe.run(on_accept=lambda spk: print(f"Welcome {spk}!"))
    """

    def __init__(self, cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
        self.cfg_path = cfg_path
        cfg = _load_cfg(cfg_path)
        self.ww_cfg = cfg["wake_word"]
        self.sp_cfg = cfg["speaker_verification"]
        log_cfg     = cfg["logging"]

        # Logging
        logging.basicConfig(
            level=getattr(logging, log_cfg["level"], logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_cfg["pipeline_log"], encoding="utf-8"),
            ]
        )
        self.log = logging.getLogger("OmniVoice")
        Path(log_cfg["pipeline_log"]).parent.mkdir(parents=True, exist_ok=True)

        # Audio stack
        self.capture  = AudioCapture(cfg_path)
        self.vad      = SileroVAD(cfg_path)
        self.nr       = NoiseReducer(cfg_path)
        self.fe       = FeatureExtractor(cfg_path)
        self.verifier = SpeakerVerifier(cfg_path)

        # ONNX wake word session (loaded lazily)
        self._ww_session: Optional[ort.InferenceSession] = None
        self.ww_threshold = self.ww_cfg["threshold"]

        self.state = State.IDLE
        self._running = False

    # ── Wake word model ─────────────────────────────────────────────────────
    def _load_ww_session(self) -> ort.InferenceSession:
        onnx_path = self.ww_cfg["onnx_path"]
        if Path(onnx_path).exists():
            self.log.info(f"Loading wake word ONNX: {onnx_path}")
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:
            self.log.warning(f"Wake word ONNX not found at {onnx_path}. "
                             f"Run: python wake_word/export.py first.")
            return None

    def _detect_wake_word(self, audio: np.ndarray) -> tuple[bool, float]:
        """Run wake word model on 1-second audio chunk."""
        if self._ww_session is None:
            return False, 0.0

        mel = self.fe.log_mel_wake_word(audio)        # [64, 101]
        inp = mel[np.newaxis, np.newaxis, :, :]       # [1, 1, 64, 101]
        logits = self._ww_session.run(
            None, {"mel_input": inp.astype(np.float32)}
        )[0]
        probs = self._softmax(logits[0])
        wake_prob = probs[1]                          # index 1 = wake word class
        return wake_prob >= self.ww_threshold, float(wake_prob)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Main loop ───────────────────────────────────────────────────────────
    def run(
        self,
        on_accept: Optional[Callable[[Optional[str], float], None]] = None,
        on_reject: Optional[Callable[[float], None]] = None,
        max_iterations: int = 0,          # 0 = run forever
    ) -> None:
        """
        Start the always-on pipeline.

        Args:
            on_accept:       callback(speaker_id, score) when speaker verified
            on_reject:       callback(score) when speaker rejected
            max_iterations:  stop after N accepted/rejected events (0 = infinite)
        """
        self._ww_session = self._load_ww_session()
        self._running = True
        iteration = 0

        self.log.info("="*55)
        self.log.info("  OmniVoice Pipeline STARTED")
        self.log.info(f"  Wake word: {self.ww_cfg['target_words']}")
        self.log.info(f"  Enrolled speakers: {list(self.verifier.enrolled.keys())}")
        self.log.info("="*55)

        # Rolling buffer for 1-second window (16000 samples @ 16kHz)
        WINDOW = 16000
        buffer = np.zeros(WINDOW, dtype=np.float32)
        chunk_size = self.capture.chunk_samples   # 480

        with self.capture:
            try:
                while self._running:
                    self.state = State.IDLE

                    # ── Step 1: VAD ─────────────────────────────────────────
                    chunk = self.capture.read()
                    is_sp, prob = self.vad.is_speech(chunk)
                    if not is_sp:
                        continue

                    self.state = State.VAD_ACTIVE
                    self.log.debug(f"VAD: speech detected (p={prob:.3f})")

                    # ── Step 2: Accumulate 1s window ────────────────────────
                    self.state = State.PREPROCESSING
                    buffer = np.roll(buffer, -chunk_size)
                    buffer[-chunk_size:] = chunk

                    audio_1s = self.nr.process(buffer.copy())

                    # ── Step 3: Wake word detection ─────────────────────────
                    self.state = State.WAKE_WORD
                    triggered, ww_score = self._detect_wake_word(audio_1s)
                    if not triggered:
                        self.log.debug(f"Wake word: not triggered (p={ww_score:.3f})")
                        continue

                    self.log.info(f"🎙  Wake word triggered! (p={ww_score:.3f})")

                    # ── Step 4: Speaker verification ────────────────────────
                    self.state = State.VERIFICATION
                    self.vad.reset_states()
                    self.capture.flush()

                    spk_id, spk_score = self.verifier.identify_live(
                        self.capture,
                        duration_s=self.sp_cfg["enroll_duration_s"],
                    )

                    if spk_id is not None:
                        self.state = State.ACCEPTED
                        self.log.info(
                            f"✓ ACCEPTED | speaker='{spk_id}' | score={spk_score:.4f}"
                        )
                        if on_accept:
                            on_accept(spk_id, spk_score)
                    else:
                        self.state = State.REJECTED
                        self.log.warning(
                            f"✗ REJECTED | unknown speaker | score={spk_score:.4f}"
                        )
                        if on_reject:
                            on_reject(spk_score)

                    iteration += 1
                    if max_iterations > 0 and iteration >= max_iterations:
                        break

            except KeyboardInterrupt:
                self.log.info("Pipeline stopped by user (Ctrl+C).")
            finally:
                self._running = False
                self.log.info("OmniVoice Pipeline STOPPED.")

    def stop(self) -> None:
        self._running = False


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = p.parse_args()

    def on_accept(spk_id, score):
        print(f"\n  🟢 Access GRANTED → {spk_id}  (score={score:.4f})\n")

    def on_reject(score):
        print(f"\n  🔴 Access DENIED   (score={score:.4f})\n")

    pipe = OmniVoicePipeline(cfg_path=args.config)
    pipe.run(on_accept=on_accept, on_reject=on_reject)
