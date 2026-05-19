"""
preprocessing/audio_capture.py
────────────────────────────────────────────────────────────────
Real-time microphone capture using sounddevice.
Fills a thread-safe ring buffer with 30ms audio chunks @ 16kHz.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import yaml
from pathlib import Path


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AudioCapture:
    """
    Always-on microphone capture.

    Usage:
        cap = AudioCapture()
        cap.start()
        chunk = cap.read()   # blocking, returns np.ndarray [N] float32
        cap.stop()
    """

    def __init__(self, cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml"):
        cfg = _load_cfg(cfg_path)["audio"]
        self.sample_rate: int = cfg["sample_rate"]          # 16000
        self.channels: int = cfg["channels"]                # 1
        self.chunk_ms: int = cfg["chunk_duration_ms"]       # 30
        self.chunk_samples: int = int(self.sample_rate * self.chunk_ms / 1000)  # 480
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._stream: sd.InputStream | None = None
        self._running: threading.Event = threading.Event()

    # ── Internal callback (called by sounddevice audio thread) ──────────────
    def _callback(self, indata: np.ndarray, frames: int,
                  time, status) -> None:
        if status:
            print(f"[AudioCapture] sounddevice status: {status}")
        # indata shape: (frames, channels) → flatten to 1-D
        chunk = indata[:, 0].copy().astype(np.float32)
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            pass   # drop oldest chunk if buffer full

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self) -> None:
        """Open microphone stream and begin capturing."""
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        print(f"[AudioCapture] Mic open — {self.sample_rate}Hz, "
              f"{self.chunk_ms}ms chunks ({self.chunk_samples} samples)")

    def read(self, timeout: float = 2.0) -> np.ndarray:
        """
        Blocking read of one audio chunk.
        Returns float32 numpy array of shape (chunk_samples,).
        Raises queue.Empty if no data arrives within `timeout` seconds.
        """
        return self._q.get(timeout=timeout)

    def read_seconds(self, duration_s: float) -> np.ndarray:
        """
        Read and concatenate enough chunks to cover `duration_s` seconds.
        """
        n_chunks = int(np.ceil(duration_s * 1000 / self.chunk_ms))
        chunks = [self.read() for _ in range(n_chunks)]
        return np.concatenate(chunks, axis=0)

    def stop(self) -> None:
        """Close the microphone stream."""
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioCapture] Mic closed.")

    def flush(self) -> None:
        """Discard all buffered chunks."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("Recording 3 seconds from microphone...")
    cap = AudioCapture()
    cap.start()
    audio = cap.read_seconds(3.0)
    cap.stop()
    print(f"Captured {len(audio)} samples ({len(audio)/16000:.2f}s) "
          f"| min={audio.min():.4f} max={audio.max():.4f}")
