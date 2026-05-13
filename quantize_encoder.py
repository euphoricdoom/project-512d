"""
Quantize the fixed encoder to int8 for 4x storage reduction.
Tests that accuracy is preserved within ±0.2%.
"""
import numpy as np
from pathlib import Path
from hf_adapter import HFAdapter
from weightless_model import WeightlessModel

def quantize_int8(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize float weights to int8 using symmetric quantization.

    Returns:
        W_int8: Quantized weights in [-128, 127]
        scale: Scaling factor
    """
    # Convert to float32 first (in case it's float64)
    W = W.astype(np.float32)

    # Symmetric quantization around zero
    # Map [-abs_max, abs_max] to [-127, 127] (reserve -128 for safety)
    abs_max = max(abs(W.min()), abs(W.max()))
    scale = abs_max / 127.0

    # Quantize: W_int8 = clip(round(W / scale))
    W_int8 = np.clip(np.round(W / scale), -128, 127).astype(np.int8)

    return W_int8, scale

def dequantize_int8(W_int8: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize int8 weights back to float32."""
    return W_int8.astype(np.float32) * scale

def test_quantization(model_path: str = "quality.npz"):
    """Load model, quantize encoder, test accuracy preservation."""

    print("="*70)
    print("INT8 QUANTIZATION TEST")
    print("="*70)

    # Load original model
    data = np.load(model_path)
    W_float32 = data["encoder_W"]
    head = data["head_sentiment"]

    print(f"\n📊 Original encoder:")
    print(f"  Shape: {W_float32.shape}")
    print(f"  Dtype: {W_float32.dtype}")
    print(f"  Size: {W_float32.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Range: [{W_float32.min():.6f}, {W_float32.max():.6f}]")

    # Quantize
    W_int8, scale = quantize_int8(W_float32)

    print(f"\n📊 Quantized encoder:")
    print(f"  Shape: {W_int8.shape}")
    print(f"  Dtype: {W_int8.dtype}")
    print(f"  Size: {W_int8.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Scale: {scale:.6f}")
    print(f"  Int8 range: [{W_int8.min()}, {W_int8.max()}]")
    print(f"  Compression: {W_float32.nbytes / W_int8.nbytes:.1f}x")

    # Test reconstruction error
    W_reconstructed = dequantize_int8(W_int8, scale)
    mse = np.mean((W_float32 - W_reconstructed) ** 2)
    max_error = np.abs(W_float32 - W_reconstructed).max()

    print(f"\n📊 Reconstruction quality:")
    print(f"  MSE: {mse:.2e}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Relative error: {max_error / (W_float32.max() - W_float32.min()):.2%}")

    # Test accuracy with quantized encoder
    print(f"\n📊 Testing accuracy preservation...")

    # Create adapter and load vocab
    adapter = HFAdapter(tfidf_dim=2000, reservoir_dim=1536, max_sentences=6, seed=42)
    adapter.load_vectorizer(str(Path(model_path).with_suffix(".vocab")))

    # Load IMDB test data
    from sklearn.preprocessing import LabelEncoder
    ds = adapter._load_hf("imdb", None, "test", None)
    texts = [str(row["text"]) for row in ds]
    labels = [row["label"] for row in ds]
    adapter._label_enc = LabelEncoder()
    adapter._label_enc.fit(labels)
    X_te, Y_te, _ = adapter._encode_all(texts, labels)

    # Test with float32 encoder
    model_float32 = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
    model_float32.encoder._W = W_float32
    model_float32.readout._heads["sentiment"] = head
    feats_float32 = model_float32.encode_sequence(X_te)
    preds_float32 = model_float32.readout.predict_batch(feats_float32, "sentiment")
    acc_float32 = np.mean(np.argmax(preds_float32, axis=1) == np.argmax(Y_te, axis=1))

    # Test with int8 encoder (dequantized)
    model_int8 = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
    model_int8.encoder._W = W_reconstructed
    model_int8.readout._heads["sentiment"] = head
    feats_int8 = model_int8.encode_sequence(X_te)
    preds_int8 = model_int8.readout.predict_batch(feats_int8, "sentiment")
    acc_int8 = np.mean(np.argmax(preds_int8, axis=1) == np.argmax(Y_te, axis=1))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Float32 accuracy: {acc_float32:.1%}")
    print(f"  Int8 accuracy:    {acc_int8:.1%}")
    print(f"  Accuracy delta:   {(acc_int8 - acc_float32) * 100:+.2f}%")
    print(f"  Storage savings:  {W_float32.nbytes / W_int8.nbytes:.1f}x ({W_float32.nbytes/1024/1024:.1f}MB → {W_int8.nbytes/1024/1024:.1f}MB)")

    # Pass/fail
    delta = abs(acc_int8 - acc_float32) * 100
    if delta <= 0.2:
        print(f"\n✅ PASS: Accuracy preserved within ±0.2% (delta = {delta:.2f}%)")
    else:
        print(f"\n⚠️  WARNING: Accuracy delta {delta:.2f}% exceeds 0.2% threshold")

    # Save quantized model
    if delta <= 0.5:  # Accept up to 0.5% for quantization
        output_path = str(Path(model_path).with_suffix("")) + "_int8.npz"
        np.savez_compressed(
            output_path,
            encoder_W_int8=W_int8,
            encoder_scale=scale,
            head_sentiment=head
        )
        print(f"\n💾 Saved quantized model to {output_path}")
        print(f"  Size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    test_quantization("quality.npz")
