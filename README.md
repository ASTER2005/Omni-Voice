# OmniVoice — Edge Voice Processing Layer

**Low-Power Edge-Based Wake Word & Speaker Verification Framework**
*Privacy-Preserving Voice-Controlled Systems — LibriSpeech Edition*

---

## 📁 Project Structure

```text
omni_voice/
├── preprocessing/          # Audio capture, VAD, noise reduction, features
├── wake_word/              # DS-CNN wake word detection (ONNX + INT8)
├── speaker_verification/   # ResNet-18 + ArcFace speaker verification
├── pipeline/               # End-to-end state machine orchestrator
├── evaluation/             # Metrics (EER, FAR, FRR) + plots
├── benchmarks/             # Per-stage latency profiling
├── tests/                  # pytest unit tests
├── models/                 # Saved model checkpoints
├── enrolled_speakers/      # Speaker embedding .npy files
└── logs/                   # Pipeline + event logs
```

---

## 🚀 Setup

### 1. Create Environment

```powershell
conda create -n omnivoice python=3.11 -y
conda activate omnivoice
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Dataset Path

Edit:

```text
pipeline/config.yaml
```

Set:

```yaml
dataset:
  root: path/to/LibriSpeech
```

---

## 🔄 Workflow

### Phase 1 — Train Wake Word Model

```powershell
python wake_word/train.py --config pipeline/config.yaml
```

### Phase 2 — Export to ONNX + INT8 Quantization

```powershell
python wake_word/export.py --config pipeline/config.yaml
```

### Phase 3 — Train Speaker Verification Model

```powershell
python speaker_verification/train.py --config pipeline/config.yaml
```

### Phase 4 — Export Speaker Model to ONNX

```powershell
python speaker_verification/export.py --config pipeline/config.yaml
```

### Phase 5 — Enroll Authorized Speaker

```powershell
python speaker_verification/manage.py --enroll --id your_name
```

### Phase 6 — Run Live Pipeline

```powershell
python pipeline/orchestrator.py --config pipeline/config.yaml
```

---

## 👤 Speaker Management

### Enroll Speaker

```powershell
python speaker_verification/manage.py --enroll --id ankan
```

### View Speaker Information

```powershell
python speaker_verification/manage.py --info --id ankan
```

### List Enrolled Speakers

```powershell
python speaker_verification/manage.py --list
```

### Reset Speaker Profile

```powershell
python speaker_verification/manage.py --reset --id ankan
```

### Secure Reset (Identity Verification Required)

```powershell
python speaker_verification/manage.py --reset --id ankan --verify-before-reset
```

### Re-Enroll Speaker

```powershell
python speaker_verification/manage.py --re-enroll --id ankan
```

### Remove All Speakers

```powershell
python speaker_verification/manage.py --reset-all
```

---

## 📊 Evaluation

### Wake Word Metrics

```powershell
python wake_word/evaluate.py
```

Measures:

* FAR (False Acceptance Rate)
* FRR (False Rejection Rate)
* EER (Equal Error Rate)
* AUC

### Speaker Verification Metrics

```powershell
python speaker_verification/evaluate.py
```

Measures:

* EER
* min-DCF

### Latency Benchmarking

```powershell
python benchmarks/profiler.py
```

### Unit Tests

```powershell
pytest tests/ -v
```

---

## 🎯 Target Performance

| Module               | Metric               | Target  |
| -------------------- | -------------------- | ------- |
| Wake Word            | FAR                  | ≤ 1%    |
| Wake Word            | FRR                  | ≤ 5%    |
| Wake Word            | Inference Latency    | ≤ 10 ms |
| Speaker Verification | EER                  | ≤ 5%    |
| Speaker Verification | min-DCF              | ≤ 0.15  |
| Speaker Verification | Verification Latency | ≤ 50 ms |

---

## 📂 Dataset

The full dataset is hosted externally due to size limitations.﻿The dataset is hosted on Kaggle due to large size (25GB).
 
**LibriSpeech ASR Corpus**

Dataset Link:🔗 https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus


https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus

### Recommended Splits

| Split           | Speakers | Purpose                       |
| --------------- | -------- | ----------------------------- |
| train-clean-360 | 921      | Training                      |
| dev-clean       | 40       | Validation & Threshold Tuning |
| test-clean      | 40       | Final Evaluation              |

---

## 🛠️ Technology Stack

### Audio Processing

* sounddevice
* librosa
* torchaudio
* noisereduce
* silero-vad

### Machine Learning

* PyTorch
* SpeechBrain
* Scikit-Learn

### Deployment

* ONNX Runtime
* INT8 Quantization

### Visualization & Analysis

* Matplotlib
* Seaborn

---

## 🔒 Features

* Wake Word Detection using DS-CNN
* Speaker Verification using ResNet-18 + ArcFace
* ONNX Runtime Inference
* INT8 Quantized Edge Deployment
* Voice Activity Detection (VAD)
* Noise Reduction Pipeline
* Real-Time Speaker Enrollment
* Secure Speaker Reset & Re-Enrollment
* End-to-End Voice Processing State Machine

---

## 📜 License

This project is intended for educational and research purposes.
