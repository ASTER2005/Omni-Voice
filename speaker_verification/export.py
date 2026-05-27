"""
speaker_verification/export.py
Export ResNet-18 speaker encoder to ONNX for fast CPU inference.
"""
import sys, yaml, torch, onnx, onnxruntime as ort, numpy as np, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from speaker_verification.model import ResNet18SpeakerEncoder


def _load_cfg(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def export(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    cfg = _load_cfg(cfg_path)
    sp = cfg["speaker_verification"]
    model_path = Path(sp["model_path"])
    onnx_path  = Path(sp["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(model_path), map_location="cpu")
    encoder = ResNet18SpeakerEncoder(embedding_dim=sp["embedding_dim"])
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    dummy = torch.randn(1, 1, 80, 300)  # [B, 1, n_mels, max_frames]

    torch.onnx.export(
        encoder, dummy, str(onnx_path),
        input_names=["mel_input"],
        output_names=["embedding"],
        dynamic_axes={"mel_input": {0: "batch_size"}},
        opset_version=17,
        export_params=True,
    )
    onnx.checker.check_model(str(onnx_path))
    print(f"[Export] Speaker ONNX saved → {onnx_path}")

    # Benchmark
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = dummy.numpy()
    for _ in range(10): sess.run(None, {"mel_input": inp})  # warm-up
    t0 = time.perf_counter()
    for _ in range(100): sess.run(None, {"mel_input": inp})
    ms = (time.perf_counter() - t0) * 10
    print(f"[Benchmark] Speaker ONNX avg latency: {ms:.2f} ms [target ≤ 50ms]  "
          f"{'✓' if ms <= 50 else '✗'}")


def load_onnx_session(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    cfg = _load_cfg(cfg_path)
    return ort.InferenceSession(
        cfg["speaker_verification"]["onnx_path"],
        providers=["CPUExecutionProvider"]
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    args = p.parse_args()
    export(cfg_path=args.config)
