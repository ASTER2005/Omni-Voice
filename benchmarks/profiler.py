"""
benchmarks/profiler.py
────────────────────────────────────────────────────────────────
Per-stage latency profiling for the full OmniVoice pipeline.
Reports: VAD, noise reduction, feature extraction, wake word ONNX,
         speaker verification ONNX.
"""
import sys, time, yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor


def _load_cfg(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _timeit(fn, n=200):
    """Run fn() n times and return average latency in ms."""
    for _ in range(10): fn()   # warm-up
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) * 1000 / n


def profile(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    cfg = _load_cfg(cfg_path)
    sr  = cfg["audio"]["sample_rate"]

    dummy_30ms  = np.random.randn(int(sr * 0.03)).astype(np.float32) * 0.05
    dummy_1s    = np.random.randn(sr).astype(np.float32) * 0.05
    dummy_4s    = np.random.randn(sr * 4).astype(np.float32) * 0.05

    nr = NoiseReducer(cfg_path)
    fe = FeatureExtractor(cfg_path)

    results = []

    # VAD (Silero)
    try:
        from preprocessing.vad import SileroVAD
        vad = SileroVAD(cfg_path)
        lat = _timeit(lambda: vad.is_speech(dummy_30ms))
        results.append(("VAD (silero-vad)", lat, "≤ 5 ms"))
    except Exception as e:
        results.append(("VAD (silero-vad)", -1, str(e)))

    # Bandpass filter
    lat = _timeit(lambda: nr.bandpass(dummy_1s))
    results.append(("Bandpass filter (scipy)", lat, "—"))

    # Spectral gate
    lat = _timeit(lambda: nr.spectral_gate(dummy_1s))
    results.append(("Spectral gate (noisereduce)", lat, "—"))

    # Full noise reduction
    lat = _timeit(lambda: nr.process(dummy_1s))
    results.append(("Noise reduction (full)", lat, "—"))

    # Log-Mel spectrogram (wake word)
    lat = _timeit(lambda: fe.log_mel_wake_word(dummy_1s))
    results.append(("Log-Mel (64-band, 1s)", lat, "—"))

    # Mel speaker feature
    lat = _timeit(lambda: fe.mel_speaker(dummy_4s))
    results.append(("Mel filterbank (80-band, 4s)", lat, "—"))

    # Wake word ONNX
    ww_onnx = cfg["wake_word"]["onnx_path"]
    if Path(ww_onnx).exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(ww_onnx, providers=["CPUExecutionProvider"])
        mel = fe.log_mel_wake_word(dummy_1s)[np.newaxis, np.newaxis].astype(np.float32)
        lat = _timeit(lambda: sess.run(None, {"mel_input": mel}))
        results.append(("Wake Word ONNX (INT8)", lat, "≤ 10 ms"))
    else:
        results.append(("Wake Word ONNX", -1, "Model not exported yet"))

    # Speaker ONNX
    sp_onnx = cfg["speaker_verification"]["onnx_path"]
    if Path(sp_onnx).exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(sp_onnx, providers=["CPUExecutionProvider"])
        mel = fe.mel_speaker(dummy_4s)
        if mel.shape[1] >= 300: mel = mel[:, :300]
        else: mel = np.pad(mel, ((0,0),(0, 300-mel.shape[1])))
        inp = mel[np.newaxis, np.newaxis].astype(np.float32)
        lat = _timeit(lambda: sess.run(None, {"mel_input": inp}))
        results.append(("Speaker ONNX", lat, "≤ 50 ms"))
    else:
        results.append(("Speaker ONNX", -1, "Model not exported yet"))

    # Print table
    print("\n" + "="*65)
    print("  OMNIVOICE PIPELINE — LATENCY BENCHMARK")
    print("="*65)
    print(f"{'Stage':<35} {'Avg (ms)':>10} {'Target':>12}")
    print("─"*65)
    total = 0
    for name, ms, target in results:
        if ms >= 0:
            ok = ""
            total += ms
            print(f"{name:<35} {ms:>9.2f}ms {target:>12}")
        else:
            print(f"{name:<35} {'N/A':>10} {target:>12}")
    print("─"*65)
    print(f"{'TOTAL (measured stages)':<35} {total:>9.2f}ms")
    print("="*65)

    # Save as CSV
    df = pd.DataFrame(results, columns=["Stage", "Latency_ms", "Target"])
    out = Path("C:/Omni_Voice/evaluation/results/latency_benchmark.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)
    print(f"\n[Profiler] Results saved → {out}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    a = p.parse_args()
    profile(cfg_path=a.config)
