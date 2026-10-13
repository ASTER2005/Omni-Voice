"""
tests/test_asr_tts.py
────────────────────────────────────────────────────────────────
Quick smoke tests for the ASR and TTS modules.
Run:
    pytest tests/test_asr_tts.py -v
Or stand-alone:
    python tests/test_asr_tts.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "C:/Omni_Voice")


# ── TTS tests ────────────────────────────────────────────────────────────────
def test_tts_pyttsx3_loads():
    """pyttsx3 should load as offline fallback."""
    from tts.speak import TTSSpeaker
    tts = TTSSpeaker(backend="pyttsx3")
    assert tts._backend == "pyttsx3"
    print("[PASS] pyttsx3 TTS loads OK")


def test_tts_pyttsx3_say(capsys=None):
    """pyttsx3 should speak without crashing (no audio assertion)."""
    from tts.speak import TTSSpeaker
    tts = TTSSpeaker(backend="pyttsx3")
    # This actually plays audio — just ensure no exception is raised
    tts.say("Testing OmniVoice text to speech.")
    print("[PASS] pyttsx3 TTS say() OK")


def test_tts_edge_loads():
    """edge-tts Communicate class should be importable."""
    from edge_tts import Communicate
    assert Communicate is not None
    print("[PASS] edge-tts imports OK")


# ── ASR tests ────────────────────────────────────────────────────────────────
def test_asr_loads():
    """StreamingASR should load Whisper model without crashing."""
    from asr.stream_asr import StreamingASR
    asr = StreamingASR(model_size="tiny", language="en")
    assert asr._model is not None
    assert asr._backend in ("faster", "openai")
    print(f"[PASS] ASR loaded — backend={asr._backend}")


def test_asr_transcribe_silence():
    """Transcribing silence should return empty string or short noise."""
    from asr.stream_asr import StreamingASR
    asr = StreamingASR(model_size="tiny", language="en")
    silence = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence
    result = asr.transcribe_once(silence)
    # Accept empty or very short garbage from silence
    assert len(result) < 100, f"Unexpected output from silence: '{result}'"
    print(f"[PASS] ASR silence test OK — output: '{result}'")


def test_asr_transcribe_tone():
    """Transcribing a 440Hz tone should not crash."""
    from asr.stream_asr import StreamingASR
    asr = StreamingASR(model_size="tiny", language="en")
    t = np.linspace(0, 3, 16000 * 3, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = asr.transcribe_once(tone)
    assert isinstance(result, str)
    print(f"[PASS] ASR tone test OK — output: '{result}'")


# ── Run standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== OmniVoice ASR + TTS Smoke Tests ===\n")
    test_tts_pyttsx3_loads()
    test_asr_loads()
    test_asr_transcribe_silence()
    test_asr_transcribe_tone()
    print("\nNote: Skipping pyttsx3 say() in CI mode (needs audio device).")
    print("\n=== All tests passed! ===\n")
