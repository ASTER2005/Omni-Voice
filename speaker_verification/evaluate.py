"""
speaker_verification/evaluate.py — EER, min-DCF, ROC/DET on test-clean.
speaker_verification/export.py  — Export encoder to ONNX.
"""
# ── evaluate.py content is below. export.py follows at the bottom.

"""
speaker_verification/evaluate.py
"""
import sys, yaml, torch, numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from speaker_verification.dataset import LibriSpeechIndex, SpeakerDataset
from speaker_verification.model import ResNet18SpeakerEncoder
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.noise_reduction import NoiseReducer
import soundfile as sf


def _load_cfg(cfg_path="C:/Omni_Voice/pipeline/config.yaml"):
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pad_mel(mel, max_frames=300):
    if mel.shape[1] >= max_frames:
        return mel[:, :max_frames]
    return np.pad(mel, ((0,0),(0, max_frames - mel.shape[1])))


def _embed_all(split, cfg_path, encoder, device, max_per_spk=5):
    cfg = _load_cfg(cfg_path)
    fe = FeatureExtractor(cfg_path)
    nr = NoiseReducer(cfg_path)
    indexer = LibriSpeechIndex(cfg_path)
    idx2files, _ = indexer.build(split)

    spk_embeddings = {}
    for spk_idx, files in tqdm(idx2files.items(), desc=f"Embedding {split}"):
        embs = []
        for fpath in files[:max_per_spk]:
            try:
                audio, _ = sf.read(fpath, dtype="float32")
                if audio.ndim > 1: audio = audio[:,0]
                audio = nr.process(audio)
                mel = _pad_mel(fe.mel_speaker(audio))
                t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = encoder(t).squeeze(0).cpu().numpy()
                embs.append(emb / (np.linalg.norm(emb) + 1e-9))
            except Exception:
                continue
        if embs:
            spk_embeddings[spk_idx] = np.stack(embs)
    return spk_embeddings


def evaluate(cfg_path="C:/Omni_Voice/pipeline/config.yaml", model_path=None):
    cfg = _load_cfg(cfg_path)
    model_path = model_path or cfg["speaker_verification"]["model_path"]
    device = torch.device("cpu")

    ckpt = torch.load(model_path, map_location=device)
    encoder = ResNet18SpeakerEncoder(embedding_dim=cfg["speaker_verification"]["embedding_dim"])
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    print(f"[SpeakerEval] Loaded {model_path}")

    spk_embs = _embed_all(cfg["dataset"]["test_split"], cfg_path, encoder, device)
    spk_ids = list(spk_embs.keys())

    scores, labels = [], []

    # Target (genuine) trials: same speaker, different utterances
    for sid in spk_ids:
        embs = spk_embs[sid]
        for i, j in combinations(range(len(embs)), 2):
            scores.append(float(np.dot(embs[i], embs[j])))
            labels.append(1)

    # Imposter trials: cross-speaker pairs
    import random
    for _ in range(len([l for l in labels if l==1])):
        s1, s2 = random.sample(spk_ids, 2)
        e1 = spk_embs[s1][0]
        e2 = spk_embs[s2][0]
        scores.append(float(np.dot(e1, e2)))
        labels.append(0)

    scores = np.array(scores)
    labels = np.array(labels)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    roc_auc = auc(fpr, tpr)

    # min-DCF
    p_target, c_miss, c_fa = 0.01, 1.0, 1.0
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = dcf.min()

    print(f"\n{'='*50}")
    print(f"  SPEAKER VERIFICATION EVALUATION")
    print(f"{'='*50}")
    print(f"  EER:     {eer*100:.2f}%   [target ≤ 5%]")
    print(f"  min-DCF: {min_dcf:.4f}   [target ≤ 0.15]")
    print(f"  AUC-ROC: {roc_auc:.4f}")
    print(f"  Threshold at EER: {thresholds[eer_idx]:.4f}")
    print(f"{'='*50}\n")

    return {"EER": eer, "min_DCF": min_dcf, "AUC": roc_auc,
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="C:/Omni_Voice/pipeline/config.yaml")
    p.add_argument("--model", default=None)
    a = p.parse_args()
    evaluate(cfg_path=a.config, model_path=a.model)
