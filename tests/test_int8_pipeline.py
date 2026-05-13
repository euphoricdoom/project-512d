"""Integration tests: QuantizedEncoder is used automatically for int8 models."""
import numpy as np
import pytest
from pathlib import Path

from int8_matmul import QuantizedEncoder
from train_hf import _install_encoder, save_model, _quantize_int8
from weightless_model import WeightlessModel, FixedEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(reservoir_dim: int = 32, input_dim: int = 16) -> WeightlessModel:
    return WeightlessModel(input_dim=input_dim, reservoir_dim=reservoir_dim,
                           output_dim=2, seed=0)


def _make_npz_int8(tmp_path, reservoir_dim=32, input_dim=16):
    rng = np.random.default_rng(1)
    W = rng.normal(size=(reservoir_dim, input_dim)).astype(np.float32) * 0.01
    W_int8, scale = _quantize_int8(W)
    path = tmp_path / "model_int8.npz"
    np.savez(path, encoder_W_int8=W_int8, encoder_scale=np.float32(scale))
    return path, W_int8, scale


def _make_npz_float(tmp_path, reservoir_dim=32, input_dim=16):
    rng = np.random.default_rng(2)
    W = rng.normal(size=(reservoir_dim, input_dim)).astype(np.float32) * 0.01
    path = tmp_path / "model_float.npz"
    np.savez(path, encoder_W=W)
    return path, W


# ---------------------------------------------------------------------------
# _install_encoder selects the right encoder
# ---------------------------------------------------------------------------

def test_install_encoder_int8_returns_quantized_encoder(tmp_path):
    path, _, _ = _make_npz_int8(tmp_path)
    model = _make_model()
    data = np.load(path)
    compute = _install_encoder(model, data)
    assert compute == "int8"
    assert isinstance(model.encoder, QuantizedEncoder)


def test_install_encoder_float_keeps_fixed_encoder(tmp_path):
    path, _ = _make_npz_float(tmp_path)
    model = _make_model()
    data = np.load(path)
    compute = _install_encoder(model, data)
    assert compute == "float32"
    assert isinstance(model.encoder, FixedEncoder)


def test_install_encoder_no_weights_raises(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, dummy=np.array([1]))
    model = _make_model()
    with pytest.raises(KeyError):
        _install_encoder(model, np.load(path))


# ---------------------------------------------------------------------------
# QuantizedEncoder produces valid output
# ---------------------------------------------------------------------------

def test_quantized_encoder_forward_pass(tmp_path):
    path, W_int8, scale = _make_npz_int8(tmp_path, reservoir_dim=32, input_dim=16)
    model = _make_model(reservoir_dim=32, input_dim=16)
    _install_encoder(model, np.load(path))

    X = np.random.default_rng(99).normal(size=(4, 16)).astype(np.float32)
    out = model.encoder.encode(X)
    assert out.shape == (4, 32)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= 1.0)  # tanh output bounded in (-1, 1)


def test_quantized_encoder_matches_float_within_tolerance(tmp_path):
    """int8 compute should match float32 to within quantization error."""
    input_dim, reservoir_dim = 64, 128
    rng = np.random.default_rng(7)
    W = rng.normal(size=(reservoir_dim, input_dim)).astype(np.float32) * 0.01
    W_int8, scale = _quantize_int8(W)

    enc_float = _make_model(reservoir_dim, input_dim)
    enc_float.encoder._W = W

    enc_int8 = _make_model(reservoir_dim, input_dim)
    enc_int8.encoder = QuantizedEncoder(W_int8, scale)

    X = rng.normal(size=(10, input_dim)).astype(np.float32)
    out_f = enc_float.encoder.encode(X)
    out_i = enc_int8.encoder.encode(X)

    mae = np.abs(out_f - out_i).mean()
    assert mae < 0.01, f"Mean absolute error {mae:.4f} exceeds tolerance"


# ---------------------------------------------------------------------------
# save_model round-trip: QuantizedEncoder weights are not re-quantized
# ---------------------------------------------------------------------------

def test_save_model_preserves_int8_weights_exactly(tmp_path):
    """Saving a QuantizedEncoder model must not dequantize→requantize."""
    input_dim, reservoir_dim = 16, 32
    W_int8_orig = np.random.default_rng(3).integers(
        -127, 128, size=(reservoir_dim, input_dim), dtype=np.int8
    )
    scale_orig = 0.001234

    model = _make_model(reservoir_dim, input_dim)
    model.encoder = QuantizedEncoder(W_int8_orig, scale_orig)

    out_path = str(tmp_path / "saved.npz")
    stub = type("Adapter", (), {
        "tfidf_dim": input_dim,
        "reservoir_dim": reservoir_dim,
        "max_sentences": 6,
        "save_vectorizer": lambda self, path: None,
    })()
    save_model(model, stub, out_path, {})

    data = np.load(out_path)
    np.testing.assert_array_equal(data["encoder_W_int8"], W_int8_orig,
                                  err_msg="int8 weights must be stored verbatim")
    assert float(data["encoder_scale"]) == pytest.approx(scale_orig, rel=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: quality_int8.npz (skipped when file not present)
# ---------------------------------------------------------------------------

QUALITY_INT8 = Path(__file__).parent.parent / "quality_int8.npz"

@pytest.mark.skipif(not QUALITY_INT8.exists(), reason="quality_int8.npz not present")
def test_quality_int8_uses_quantized_encoder():
    """Loading quality_int8.npz must install a QuantizedEncoder, not dequantize."""
    data = np.load(QUALITY_INT8)
    assert "encoder_W_int8" in data.files, "quality_int8.npz missing int8 weights"

    model = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
    compute = _install_encoder(model, data)

    assert compute == "int8", f"Expected int8 compute, got {compute!r}"
    assert isinstance(model.encoder, QuantizedEncoder), (
        f"Expected QuantizedEncoder, got {type(model.encoder).__name__}"
    )
    assert model.encoder._W_int8.dtype == np.int8
    assert model.encoder.reservoir_dim == 1536
    assert model.encoder.input_dim == 2000


@pytest.mark.skipif(not QUALITY_INT8.exists(), reason="quality_int8.npz not present")
def test_quality_int8_forward_pass():
    """End-to-end: QuantizedEncoder loaded from quality_int8.npz produces finite output."""
    data = np.load(QUALITY_INT8)
    model = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
    _install_encoder(model, data)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 6, 2000)).astype(np.float32)
    feats = model.encode_sequence(X)

    assert feats.shape == (3, 1536 * 6 + 3)
    assert np.all(np.isfinite(feats))
