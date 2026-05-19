"""tests/test_speaker_verification.py — Unit tests for speaker models."""
import sys, pytest, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_verification.model import ResNet18SpeakerEncoder, ArcFaceHead


@pytest.fixture
def encoder():
    return ResNet18SpeakerEncoder(embedding_dim=512)

@pytest.fixture
def arc_head():
    return ArcFaceHead(embedding_dim=512, num_classes=921)


def test_encoder_output_shape(encoder):
    x = torch.randn(4, 1, 80, 300)
    emb = encoder(x)
    assert emb.shape == (4, 512)

def test_encoder_l2_normalised(encoder):
    x = torch.randn(4, 1, 80, 300)
    emb = encoder(x)
    norms = torch.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

def test_arcface_output_shape(encoder, arc_head):
    x = torch.randn(4, 1, 80, 300)
    emb = encoder(x)
    labels = torch.randint(0, 921, (4,))
    logits = arc_head(emb, labels)
    assert logits.shape == (4, 921)

def test_arcface_inference_mode(encoder, arc_head):
    x = torch.randn(2, 1, 80, 300)
    emb = encoder(x)
    logits = arc_head(emb, labels=None)   # no labels → inference
    assert logits.shape == (2, 921)

def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(np.dot(a, b) - 1.0) < 1e-6
    assert abs(np.dot(a, c) - 0.0) < 1e-6

def test_encoder_no_nan(encoder):
    x = torch.randn(4, 1, 80, 300)
    emb = encoder(x)
    assert not torch.isnan(emb).any()
