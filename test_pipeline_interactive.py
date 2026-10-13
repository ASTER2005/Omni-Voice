"""
test_pipeline_interactive.py
────────────────────────────────────────────────────────────────
Interactive test runner for OmniVoice components:
  1. Test Text-to-Speech (TTS)
  2. Test Speech Recognition (ASR)
  3. Test Speaker Verification (Ankan vs others)
  4. Test End-to-End V2 Pipeline (Wake Word / Speech -> Verify -> ASR -> TTS)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_tts():
    print("\n--- [1] Testing TTS (Text-to-Speech) ---")
    from tts.speak import TTSSpeaker
    try:
        speaker = TTSSpeaker()
        print(f"Using TTS backend: {speaker._backend}")
        test_phrase = "Hello Ankan! OmniVoice text to speech is functioning properly."
        print(f"Speaking: '{test_phrase}'...")
        speaker.say(test_phrase)
        print("✓ TTS playback finished successfully.")
    except Exception as e:
        print(f"✗ TTS test error: {e}")


def test_asr():
    print("\n--- [2] Testing ASR (Speech-to-Text) ---")
    from asr.stream_asr import StreamingASR
    from preprocessing.audio_capture import AudioCapture
    from preprocessing.vad import SileroVAD

    cfg_path = "pipeline/config.yaml"
    asr = StreamingASR(model_size="base", language="en")
    cap = AudioCapture(cfg_path)
    vad = SileroVAD(cfg_path)

    print("Opening mic. Please speak a sentence (e.g. 'What is the weather today?')...")
    with cap:
        buffer = []
        silence_count = 0
        recording = False
        print("Listening for your voice...")
        while True:
            chunk = cap.read()
            is_speech, prob = vad.is_speech(chunk)
            if is_speech:
                if not recording:
                    print(f"🎙 Voice detected! Recording...")
                    recording = True
                buffer.append(chunk)
                silence_count = 0
            else:
                if recording:
                    silence_count += 1
                    buffer.append(chunk)
                    if silence_count >= 15:  # ~450ms silence
                        print("Silence detected. Transcribing...")
                        break

    import numpy as np
    audio = np.concatenate(buffer, axis=0)
    transcription = asr.transcribe_once(audio)
    print(f"\n✓ ASR Transcribed: \"{transcription}\"\n")


def test_speaker_verification():
    print("\n--- [3] Testing Speaker Verification ---")
    from speaker_verification.verify import SpeakerVerifier
    from preprocessing.audio_capture import AudioCapture

    verifier = SpeakerVerifier("pipeline/config.yaml")
    print(f"Loaded enrolled speakers: {list(verifier.enrolled.keys())}")
    print(f"Verification threshold: {verifier.threshold}")
    print("Speak for 4 seconds when prompted to verify identity...")

    with AudioCapture("pipeline/config.yaml") as cap:
        spk_id, score = verifier.identify_live(cap, duration_s=4.0)
        if spk_id:
            print(f"✓ ACCEPTED: Speaker identified as '{spk_id}' (similarity score: {score:.4f})")
        else:
            print(f"✗ REJECTED: Voice not recognized (best similarity score: {score:.4f} < {verifier.threshold})")


def test_full_v2_direct():
    print("\n--- [4] Testing Full V2 Cycle (Direct / Skip Wake Word) ---")
    from pipeline.orchestrator_v2 import OmniVoicePipelineV2

    def sample_command(spk, text):
        print(f"\n[Command Handler] Speaker: '{spk}', Command: '{text}'")
        if "time" in text.lower():
            import datetime
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
        return f"Hello {spk}, I processed your request: {text}"

    pipe = OmniVoicePipelineV2(
        cfg_path="pipeline/config.yaml",
        skip_wake_word=True,
        on_command=sample_command,
    )
    print("\nStarting 1 interactive session:")
    print("1. Speak to verify your identity.")
    print("2. Listen for the TTS greeting.")
    print("3. Speak your command (e.g. 'What time is it?').")
    print("4. Listen for the TTS answer.")
    pipe.run(max_sessions=1)
    print("✓ Full interactive cycle test complete.")


def main():
    print("=" * 60)
    print("        OmniVoice V2 Interactive Test Menu")
    print("=" * 60)
    print("1. Test Text-to-Speech (TTS)")
    print("2. Test Speech Recognition (ASR)")
    print("3. Test Speaker Verification")
    print("4. Test Full V2 Cycle (Direct: Verify → Greet → ASR → Reply)")
    print("5. Run Full V2 Pipeline (Standard: Wake Word 'omni' → ...)")
    print("=" * 60)

    choice = input("Enter option [1-5] (or press Enter for 4): ").strip()
    if choice == "1":
        test_tts()
    elif choice == "2":
        test_asr()
    elif choice == "3":
        test_speaker_verification()
    elif choice == "4" or choice == "":
        test_full_v2_direct()
    elif choice == "5":
        from pipeline.orchestrator_v2 import OmniVoicePipelineV2
        pipe = OmniVoicePipelineV2(cfg_path="pipeline/config.yaml", skip_wake_word=False)
        pipe.run(max_sessions=1)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
