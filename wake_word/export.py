"""
wake_word/export.py
────────────────────────────────────────────────────────────────
Export trained WakeWordCNN to ONNX + INT8 quantized ONNX.

Steps:
  1. Load best checkpoint
  2. Export to float32 ONNX
  3. Apply PyTorch static INT8 quantization → INT8 ONNX
  4. Benchmark both with onnxruntime
"""

import sys
import yaml
import time
import torch
import torch.quantization as quant
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from wake_word.model import WakeWordCNN


def _load_cfg(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export(cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml") -> None:
    cfg = _load_cfg(cfg_path)
    model_path = Path(cfg["wake_word"]["model_path"])
    onnx_path  = Path(cfg["wake_word"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 2)
    model = WakeWordCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Export] Loaded checkpoint: {model_path}")

    # ── Dummy input ────────────────────────────────────────────────────────
    dummy = torch.randn(1, 1, 64, 101)

    # ── Float32 ONNX ───────────────────────────────────────────────────────
    fp32_path = onnx_path.parent / "wake_word_fp32.onnx"
    torch.onnx.export(
        model, dummy, str(fp32_path),
        input_names=["mel_input"],
        output_names=["logits"],
        dynamic_axes={"mel_input": {0: "batch_size"}},
        opset_version=17,
        export_params=True,
    )
    onnx.checker.check_model(str(fp32_path))
    print(f"[Export] Float32 ONNX saved → {fp32_path}")

    # ── INT8 Static Quantization ───────────────────────────────────────────
    model_q = WakeWordCNN(num_classes=num_classes)
    model_q.load_state_dict(ckpt["model_state"])
    model_q.eval()

    # Fuse conv-bn-relu patterns
    model_q = torch.quantization.fuse_modules(
        model_q,
        [
            ["input_conv.0", "input_conv.1", "input_conv.2"],
        ],
        inplace=False,
    )
    model_q.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model_q, inplace=True)

    # Calibration with random data (replace with real calibration data for best results)
    print("[Export] Calibrating quantization...")
    with torch.no_grad():
        for _ in range(100):
            calibration_input = torch.randn(8, 1, 64, 101)
            model_q(calibration_input)

    quant.convert(model_q, inplace=True)
    print("[Export] INT8 quantization complete.")

    # Export quantized model to ONNX
    torch.onnx.export(
        model_q, dummy, str(onnx_path),
        input_names=["mel_input"],
        output_names=["logits"],
        dynamic_axes={"mel_input": {0: "batch_size"}},
        opset_version=17,
        export_params=True,
    )
    print(f"[Export] INT8 ONNX saved → {onnx_path}")

    # ── Benchmark ─────────────────────────────────────────────────────────
    _benchmark_onnx(str(fp32_path), dummy.numpy(), "FP32")
    _benchmark_onnx(str(onnx_path), dummy.numpy(), "INT8")


def _benchmark_onnx(onnx_path: str, input_data: np.ndarray, tag: str) -> None:
    """Measure average inference latency with onnxruntime."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    n_runs = 200

    # Warm-up
    for _ in range(10):
        sess.run(None, {input_name: input_data})

    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: input_data})
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_runs

    print(f"[Benchmark] {tag:5s} ONNX | avg latency: {elapsed_ms:.2f} ms "
          f"[target ≤ 10ms]  {'✓' if elapsed_ms <= 10 else '✗'}")


def load_onnx_session(
    cfg_path: str = "C:/Omni_Voice/pipeline/config.yaml",
) -> ort.InferenceSession:
    """Load the INT8 ONNX model for runtime inference."""
    cfg = _load_cfg(cfg_path)
    onnx_path = cfg["wake_word"]["onnx_path"]
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = parser.parse_args()
    export(cfg_path=args.config)
