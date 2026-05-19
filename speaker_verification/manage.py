"""
speaker_verification/manage.py
────────────────────────────────────────────────────────────────
Speaker Management CLI:
  --enroll  --id <name>    Live mic enrollment
  --reset   --id <name>    Delete one speaker profile
  --reset-all              Wipe entire enrolled DB
  --re-enroll --id <name>  Reset + re-enroll
  --list                   Show all enrolled speakers + dates
  --info    --id <name>    Embedding stats for a speaker
  --verify-before-reset    Require live verification before deletion
"""

import sys, os, yaml, argparse, datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_cfg(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _log(msg, cfg):
    p = Path(cfg["logging"]["speaker_events_log"])
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    p.open("a").write(f"[{ts}] {msg}\n")
    print(f"[Log] {msg}")

def _enrolled_dir(cfg):
    return Path(cfg["speaker_verification"]["enrolled_dir"])


def cmd_enroll(speaker_id, cfg_path):
    from speaker_verification.enroll import enroll
    path = enroll(speaker_id=speaker_id, cfg_path=cfg_path)
    _log(f"ENROLL speaker_id='{speaker_id}' path='{path}'", _load_cfg(cfg_path))


def cmd_reset(speaker_id, cfg_path, verify_first=False):
    cfg = _load_cfg(cfg_path)
    target = _enrolled_dir(cfg) / f"{speaker_id}.npy"
    if not target.exists():
        print(f"[Manage] Speaker '{speaker_id}' not enrolled.")
        return False

    if verify_first:
        from speaker_verification.verify import SpeakerVerifier
        from preprocessing.audio_capture import AudioCapture
        v = SpeakerVerifier(cfg_path)
        with AudioCapture(cfg_path) as cap:
            ok, score = v.verify_live(speaker_id, cap)
        if not ok:
            print(f"[Manage] Verification failed (score={score:.4f}). Aborted.")
            return False

    if input(f"  Delete '{speaker_id}'? [y/N]: ").strip().lower() != "y":
        print("[Manage] Cancelled.")
        return False

    os.remove(str(target))
    _log(f"RESET speaker_id='{speaker_id}'", cfg)
    print(f"[Manage] Speaker '{speaker_id}' removed.")
    return True


def cmd_reset_all(cfg_path):
    cfg = _load_cfg(cfg_path)
    files = list(_enrolled_dir(cfg).glob("*.npy"))
    if not files:
        print("[Manage] No enrolled speakers.")
        return
    print(f"[Manage] {len(files)} enrolled speaker(s): {[f.stem for f in files]}")
    if input("  Delete ALL? [y/N]: ").strip().lower() != "y":
        return
    for f in files:
        os.remove(str(f))
    _log(f"RESET_ALL {[f.stem for f in files]}", cfg)
    print(f"[Manage] All {len(files)} speaker(s) removed.")


def cmd_re_enroll(speaker_id, cfg_path, verify_first=False):
    cmd_reset(speaker_id, cfg_path, verify_first)
    cmd_enroll(speaker_id, cfg_path)


def cmd_list(cfg_path):
    cfg = _load_cfg(cfg_path)
    files = sorted(_enrolled_dir(cfg).glob("*.npy"))
    if not files:
        print("[Manage] No enrolled speakers.")
        return
    print(f"\n{'Speaker':<25} {'Enrolled (UTC)':<25} {'Dim'}")
    print("─" * 60)
    for f in files:
        ts = datetime.datetime.utcfromtimestamp(f.stat().st_mtime)
        emb = np.load(str(f))
        print(f"{f.stem:<25} {ts.strftime('%Y-%m-%d %H:%M:%S'):<25} {emb.shape[0]}")
    print(f"\nTotal: {len(files)} speaker(s)")


def cmd_info(speaker_id, cfg_path):
    cfg = _load_cfg(cfg_path)
    t = _enrolled_dir(cfg) / f"{speaker_id}.npy"
    if not t.exists():
        print(f"[Manage] '{speaker_id}' not enrolled.")
        return
    emb = np.load(str(t))
    ts = datetime.datetime.utcfromtimestamp(t.stat().st_mtime)
    print(f"\nSpeaker:  {speaker_id}")
    print(f"Enrolled: {ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Emb dim:  {emb.shape[0]}  |  L2 norm: {np.linalg.norm(emb):.6f}")
    print(f"Mean/Std: {emb.mean():.6f} / {emb.std():.6f}")
    print(f"Min/Max:  {emb.min():.6f} / {emb.max():.6f}")


def main():
    p = argparse.ArgumentParser(description="OmniVoice Speaker Management")
    p.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    p.add_argument("--id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--enroll",    action="store_true")
    g.add_argument("--reset",     action="store_true")
    g.add_argument("--reset-all", action="store_true", dest="reset_all")
    g.add_argument("--re-enroll", action="store_true", dest="re_enroll")
    g.add_argument("--list",      action="store_true")
    g.add_argument("--info",      action="store_true")
    p.add_argument("--verify-before-reset", action="store_true")
    a = p.parse_args()

    if a.enroll:      cmd_enroll(a.id, a.config)
    elif a.reset:     cmd_reset(a.id, a.config, a.verify_before_reset)
    elif a.reset_all: cmd_reset_all(a.config)
    elif a.re_enroll: cmd_re_enroll(a.id, a.config, a.verify_before_reset)
    elif a.list:      cmd_list(a.config)
    elif a.info:      cmd_info(a.id, a.config)

if __name__ == "__main__":
    main()
