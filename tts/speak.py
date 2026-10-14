"""
tts/speak.py
────────────────────────────────────────────────────────────────
Text-to-Speech module for OmniVoice.

Backend priority:
  1. edge-tts  — Microsoft neural voices (online, high quality, free)
  2. pyttsx3   — Fully offline system TTS (SAPI on Windows)

Usage:
    speaker = TTSSpeaker()
    speaker.say("Hello Ankan, how can I help you?")
    # OR async:
    await speaker.say_async("Hello!")
"""

import sys
import asyncio
import threading
import tempfile
import os
from pathlib import Path
from typing import Optional


# ── Backend detection ────────────────────────────────────────────────────────
def _try_edge_tts():
    """Check if edge-tts is available."""
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _try_pyttsx3():
    """Check if pyttsx3 is available."""
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False


class TTSSpeaker:
    """
    Text-to-Speech engine for OmniVoice.

    Picks the best available backend automatically:
      - edge-tts  (online, neural, high-quality Microsoft voices)
      - pyttsx3   (offline system voices — always available on Windows)

    Args:
        voice:       Voice name string.
                     edge-tts examples: 'en-IN-NeerjaNeural', 'en-US-JennyNeural'
                     pyttsx3: 'HKEY_LOCAL_MACHINE\\...' (auto-selected if None)
        rate:        Speaking rate (words per minute) for pyttsx3 (default: 175)
        volume:      Volume 0.0–1.0 for pyttsx3 (default: 1.0)
        backend:     Force 'edge' or 'pyttsx3'. Auto-detect if None.
    """

    EDGE_VOICE_DEFAULT = "en-IN-NeerjaNeural"   # Indian English, female, neural

    def __init__(
        self,
        voice: Optional[str] = None,
        rate: int = 175,
        volume: float = 1.0,
        backend: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._backend: str = ""
        self._engine = None   # pyttsx3 engine

        # Voice selection
        self._edge_voice = voice or self.EDGE_VOICE_DEFAULT
        self._pyttsx_voice = voice  # None = system default

        # Select backend
        if backend == "edge" or (backend is None and _try_edge_tts()):
            self._backend = "edge"
            print(f"[TTS] Backend: edge-tts | voice: {self._edge_voice}")
        elif backend == "pyttsx3" or (backend is None and _try_pyttsx3()):
            self._backend = "pyttsx3"
            self._engine = self._init_pyttsx3(rate, volume)
        else:
            raise ImportError(
                "[TTS] No TTS backend available!\n"
                "  Install:  pip install edge-tts\n"
                "        OR  pip install pyttsx3"
            )

    def _init_pyttsx3(self, rate: int, volume: float):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        if self._pyttsx_voice:
            engine.setProperty("voice", self._pyttsx_voice)
        else:
            # Pick first available voice
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
                print(f"[TTS] Backend: pyttsx3 | voice: {voices[0].name}")
        return engine

    # ── Sync speak ───────────────────────────────────────────────────────────
    def say(self, text: str) -> None:
        """
        Speak text synchronously (blocks until audio finishes).

        Args:
            text: The text to speak.
        """
        if not text or not text.strip():
            return

        with self._lock:
            if self._backend == "edge":
                try:
                    self._say_edge_sync(text)
                except Exception as e:
                    print(f"\n[TTS] ⚠ edge-tts error ({type(e).__name__}: {e}).\n"
                          f"[TTS] Falling back to offline system voice (pyttsx3)...")
                    if self._engine is None:
                        self._engine = self._init_pyttsx3(rate=175, volume=1.0)
                    self._backend = "pyttsx3"
                    self._say_pyttsx3(text)
            else:
                self._say_pyttsx3(text)

    def _say_pyttsx3(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def _say_edge_sync(self, text: str) -> None:
        """Run edge-tts in a blocking manner via asyncio."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context; run in a thread
                future = asyncio.run_coroutine_threadsafe(
                    self._say_edge_async(text), loop
                )
                future.result(timeout=30)
            else:
                loop.run_until_complete(self._say_edge_async(text))
        except RuntimeError:
            # No current event loop — create one
            asyncio.run(self._say_edge_async(text))

    async def _say_edge_async(self, text: str) -> None:
        """Async edge-tts speak using a temp file + sounddevice playback."""
        import edge_tts
        import sounddevice as sd
        import soundfile as sf

        communicate = edge_tts.Communicate(text, self._edge_voice)

        # Write audio to a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await communicate.save(tmp_path)
            # Read and play
            data, samplerate = sf.read(tmp_path, dtype="float32")
            if data.ndim == 1:
                data = data[:, None]
            sd.play(data, samplerate=samplerate)
            sd.wait()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Async speak ──────────────────────────────────────────────────────────
    async def say_async(self, text: str) -> None:
        """
        Speak text asynchronously (for use in async contexts).

        Args:
            text: The text to speak.
        """
        if not text or not text.strip():
            return

        if self._backend == "edge":
            await self._say_edge_async(text)
        else:
            # pyttsx3 is not async-native — run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._say_pyttsx3, text)

    # ── Non-blocking speak (fire-and-forget thread) ──────────────────────────
    def say_nonblocking(self, text: str) -> threading.Thread:
        """
        Speak in a background thread so the caller is not blocked.

        Returns the thread so the caller can join() if needed.
        """
        t = threading.Thread(target=self.say, args=(text,), daemon=True)
        t.start()
        return t

    # ── List available edge-tts voices ───────────────────────────────────────
    @staticmethod
    async def list_edge_voices() -> list[dict]:
        """Return list of available edge-tts voices."""
        import edge_tts
        voices = await edge_tts.list_voices()
        return voices

    def print_voices(self) -> None:
        """Print available voices for the current backend."""
        if self._backend == "edge":
            voices = asyncio.run(self.list_edge_voices())
            en_voices = [v for v in voices if v["Locale"].startswith("en")]
            print("\n[TTS] Available English edge-tts voices:")
            for v in en_voices:
                print(f"  {v['ShortName']:35s} | {v['Gender']:6s} | {v['Locale']}")
        else:
            voices = self._engine.getProperty("voices")
            print("\n[TTS] Available pyttsx3 voices:")
            for v in voices:
                print(f"  {v.name:40s} | {v.id}")


# ── Standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OmniVoice TTS demo")
    p.add_argument("--text",    default="Hello! I am Omni Voice. How can I help you?")
    p.add_argument("--voice",   default=None, help="Voice name (edge-tts or pyttsx3)")
    p.add_argument("--backend", default=None, choices=["edge", "pyttsx3"])
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args()

    tts = TTSSpeaker(voice=args.voice, backend=args.backend)

    if args.list_voices:
        tts.print_voices()
    else:
        print(f"[TTS] Speaking: '{args.text}'")
        tts.say(args.text)
        print("[TTS] Done.")
