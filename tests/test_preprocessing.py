"""tests/test_preprocessing.py — Unit tests for preprocessing modules."""
import sys, pytest, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.noise_reduction import NoiseReducer
from preprocessing.feature_extraction import FeatureExtractor


@pytest.fixture
def dummy_audio():
    return np.random.randn(16000).astype(np.float32) * 0.05

@pytest.fixture
def nr():
    return NoiseReducer()

@pytest.fixture
def fe():
    return FeatureExtractor()


def test_bandpass_shape(nr, dummy_audio):
    out = nr.bandpass(dummy_audio)
    assert out.shape == dummy_audio.shape
    assert out.dtype == np.float32

def test_spectral_gate_shape(nr, dummy_audio):
    out = nr.spectral_gate(dummy_audio)
    assert out.shape == dummy_audio.shape

def test_full_pipeline_shape(nr, dummy_audio):
    out = nr.process(dummy_audio)
    assert out.shape == dummy_audio.shape
    assert out.dtype == np.float32

def test_snr_mixing(nr, dummy_audio):
    noise = np.random.randn(16000).astype(np.float32) * 0.02
    noisy = nr.add_noise_snr(dummy_audio, noise, snr_db=10.0)
    assert noisy.shape == dummy_audio.shape

def test_log_mel_wake_word_shape(fe, dummy_audio):
    mel = fe.log_mel_wake_word(dummy_audio)
    assert mel.shape == (64, 101)
    assert mel.dtype == np.float32

def test_mel_speaker_shape(fe, dummy_audio):
    mel = fe.mel_speaker(dummy_audio)
    assert mel.shape[0] == 80

def test_mfcc_shape(fe, dummy_audio):
    mfcc = fe.mfcc(dummy_audio, n_mfcc=40)
    assert mfcc.shape[0] == 40

def test_spec_augment_shape(fe, dummy_audio):
    mel = fe.log_mel_wake_word(dummy_audio)
    aug = fe.spec_augment(mel)
    assert aug.shape == mel.shape

def test_pad_or_trim():
    arr = np.ones(8000, dtype=np.float32)
    assert FeatureExtractor._pad_or_trim(arr, 16000).shape == (16000,)
    assert FeatureExtractor._pad_or_trim(arr, 4000).shape == (4000,)
