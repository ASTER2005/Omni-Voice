"""
run.py
────────────────────────────────────────────────────────────────
Direct 2-Step OmniVoice Entrypoint:
  1. Register your voice (Enrollment)
  2. Authenticate & Talk (Voice Verification + ASR + Voice Reply)
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def enroll_voice():
    name = input("\nEnter your name [default: Ankan]: ").strip() or "Ankan"
    print(f"\nStarting voice registration for '{name}'...")
    print("You will be asked to speak clearly into your microphone.\n")
    cmd = [sys.executable, str(ROOT / "speaker_verification" / "enroll.py"), "--id", name]
    subprocess.run(cmd)


def authenticate_and_talk():
    print("\nStarting voice authentication...")
    print("1. Speak into the microphone so the AI can verify your voice.")
    print("2. If accepted, the AI will greet you and listen to your command.")
    print("3. The AI will reply back with speech.")
    print("4. If somebody else speaks, access will be denied.\n")
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "orchestrator_v2.py"),
        "--skip-wake-word",
    ]
    subprocess.run(cmd)


def main():
    print("\n" + "=" * 55)
    print("             OmniVoice Voice Assistant")
    print("=" * 55)
    print("1. Register my voice")
    print("2. Authenticate & Talk (Voice verify + Speech reply)")
    print("=" * 55)

    choice = input("Select [1 or 2]: ").strip()
    if choice == "1":
        enroll_voice()
    elif choice == "2":
        authenticate_and_talk()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")


if __name__ == "__main__":
    main()
