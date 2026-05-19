"""tests/test_wake_word.py — Unit tests for wake word model."""
import sys, pytest, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_word.model import WakeWordCNN


@pytest.fixture
def model():
    return WakeWordCNN(num_classes=2, n_mels=64)

def test_model_output_shape(model):
    x = torch.randn(4, 1, 64, 101)
    out = model(x)
    assert out.shape == (4, 2)

def test_model_probabilities(model):
    x = torch.randn(2, 1, 64, 101)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

def test_parameter_count(model):
    params = model.count_parameters()
    assert params < 500_000, f"Model too large: {params:,} params"
    print(f"  Parameters: {params:,}")

def test_forward_no_nan(model):
    x = torch.randn(8, 1, 64, 101)
    out = model(x)
    assert not torch.isnan(out).any()

def test_multi_class():
    m = WakeWordCNN(num_classes=3)
    x = torch.randn(2, 1, 64, 101)
    out = m(x)
    assert out.shape == (2, 3)
