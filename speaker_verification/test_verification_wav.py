"""
speaker_verification/test_verification_wav.py
────────────────────────────────────────────────────────────────
Offline WAV/FLAC speaker verification:
  1. Load enrolled speaker profiles
  2. Load query audio file (WAV/FLAC)
  3. Preprocess and extract speaker embedding (using PyTorch or ONNX)
  4. Compare with enrolled profile and accept/reject
"""

import sys
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_verification.verify import SpeakerVerifier


def main():
    parser = argparse.ArgumentParser(description="Offline Speaker Verification from Audio File")
    parser.add_argument("--id", required=True, help="Enrolled speaker ID (e.g., Ankan)")
    parser.add_argument("--wav", required=True, help="Path to WAV or FLAC file to verify")
    parser.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model instead of PyTorch")
    args = parser.parse_args()

    # Load verifier
    print(f"[OfflineVerify] Initializing verifier (onnx={args.onnx})...")
    verifier = SpeakerVerifier(args.config, use_onnx=args.onnx)

    # Check if speaker exists
    if args.id not in verifier.enrolled:
        print(f"[OfflineVerify] Error: Speaker '{args.id}' is not enrolled!")
        print(f"[OfflineVerify] Enrolled speakers: {list(verifier.enrolled.keys())}")
        sys.exit(1)

    # Load audio file
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"[OfflineVerify] Error: Audio file '{args.wav}' not found!")
        sys.exit(1)

    print(f"[OfflineVerify] Reading {wav_path.name}...")
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # Take first channel if stereo

    # Resample if sample rate doesn't match config
    expected_sr = verifier.fe.sr
    if sr != expected_sr:
        print(f"[OfflineVerify] Warning: Sample rate mismatch ({sr} Hz vs expected {expected_sr} Hz). Resampling...")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr)

    # Apply noise reduction
    print("[OfflineVerify] Applying noise reduction...")
    audio_processed = verifier.nr.process(audio)

    # Extract embedding
    print("[OfflineVerify] Extracting embedding...")
    emb = verifier.embed(audio_processed)

    # Verify similarity
    print(f"[OfflineVerify] Calculating similarity for speaker '{args.id}'...")
    score = verifier.cosine_similarity(emb, args.id)
    accepted = score >= verifier.threshold

    status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
    print(f"\n==========================================")
    print(f"  VERIFICATION RESULTS: {args.id}")
    print(f"==========================================")
    print(f"  Status:    {status}")
    print(f"  Score:     {score:.4f}")
    print(f"  Threshold: {verifier.threshold:.4f}")
    print(f"  Backend:   {'ONNX' if args.onnx else verifier.backend}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
