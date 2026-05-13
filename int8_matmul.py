"""
Int8 matrix multiplication for NPU-native inference.

Implements int8 × int8 → int32 matmul with proper scaling.
No external dependencies beyond numpy.
"""
import numpy as np


def quantize_activation(X: np.ndarray, scale: float = None) -> tuple[np.ndarray, float]:
    """Quantize activation tensor to int8.

    Args:
        X: Input tensor (float32)
        scale: Optional pre-computed scale

    Returns:
        X_int8: Quantized tensor
        scale: Scale factor
    """
    X = X.astype(np.float32)

    if scale is None:
        # Symmetric quantization
        abs_max = max(abs(X.min()), abs(X.max()))
        if abs_max == 0:
            abs_max = 1.0  # Avoid division by zero
        scale = abs_max / 127.0

    X_int8 = np.clip(np.round(X / scale), -128, 127).astype(np.int8)
    return X_int8, scale


def int8_matmul(X_int8: np.ndarray, W_int8: np.ndarray,
                scale_x: float, scale_w: float) -> np.ndarray:
    """Compute int8 × int8 matmul with scaling.

    Equivalent to: X @ W.T in float32, but using int8 compute.

    Args:
        X_int8: Input activations (N, D_in) as int8
        W_int8: Weights (D_out, D_in) as int8
        scale_x: Scale factor for X
        scale_w: Scale factor for W

    Returns:
        Result in float32 (N, D_out)
    """
    # int8 × int8 → int32 (this is what NPUs accelerate)
    result_int32 = X_int8.astype(np.int32) @ W_int8.T.astype(np.int32)

    # Scale back to float32
    scale_total = scale_x * scale_w
    result_float32 = result_int32.astype(np.float32) * scale_total

    return result_float32


class QuantizedEncoder:
    """Fixed encoder that computes in int8.

    Drop-in replacement for FixedEncoder with int8 inference.
    """

    def __init__(self, W_int8: np.ndarray, scale_w: float):
        """Initialize with pre-quantized weights.

        Args:
            W_int8: Encoder weights (reservoir_dim, input_dim) as int8
            scale_w: Scale factor for weights
        """
        self._W_int8 = W_int8.astype(np.int8)
        self._scale_w = float(scale_w)
        self.reservoir_dim, self.input_dim = W_int8.shape

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input using int8 matmul.

        Args:
            X: Input (batch, input_dim) or (input_dim,) as float32

        Returns:
            Encoded output (batch, reservoir_dim) as float32
        """
        # Handle single vector
        if X.ndim == 1:
            X = X[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        # Quantize input activations
        X_int8, scale_x = quantize_activation(X)

        # int8 × int8 matmul
        result = int8_matmul(X_int8, self._W_int8, scale_x, self._scale_w)

        # Apply activation (tanh)
        result = np.tanh(result)

        if squeeze:
            result = result[0]

        return result.astype(np.float32)

    @property
    def _W(self) -> np.ndarray:
        """Dequantize weights for compatibility."""
        return self._W_int8.astype(np.float32) * self._scale_w

    def memory_footprint_mb(self) -> float:
        """Return memory footprint in MB."""
        weights_mb = self._W_int8.nbytes / 1024 / 1024
        scale_mb = 8 / 1024 / 1024  # float64
        return weights_mb + scale_mb


def test_int8_matmul():
    """Validate int8 matmul against float32 baseline."""
    print("="*70)
    print("INT8 MATMUL VALIDATION")
    print("="*70)

    # Test parameters
    batch_size = 100
    input_dim = 2000
    reservoir_dim = 1536

    # Create random test data
    np.random.seed(42)
    W_float = np.random.randn(reservoir_dim, input_dim).astype(np.float32) * 0.01
    X_float = np.random.randn(batch_size, input_dim).astype(np.float32)

    # Float32 baseline
    result_float = np.tanh(X_float @ W_float.T)

    # Quantize weights
    W_int8, scale_w = quantize_activation(W_float)

    # Int8 inference
    encoder = QuantizedEncoder(W_int8, scale_w)
    result_int8 = encoder.encode(X_float)

    # Compare
    mse = np.mean((result_float - result_int8) ** 2)
    max_error = np.abs(result_float - result_int8).max()
    mean_abs_error = np.abs(result_float - result_int8).mean()

    print(f"\n📊 Numerical accuracy:")
    print(f"  MSE:            {mse:.2e}")
    print(f"  Max error:      {max_error:.6f}")
    print(f"  Mean abs error: {mean_abs_error:.6f}")

    # Memory comparison
    mem_float = W_float.nbytes / 1024 / 1024
    mem_int8 = encoder.memory_footprint_mb()

    print(f"\n💾 Memory footprint:")
    print(f"  Float32: {mem_float:.2f} MB")
    print(f"  Int8:    {mem_int8:.2f} MB")
    print(f"  Savings: {mem_float / mem_int8:.1f}x")

    # Check if acceptable
    if mean_abs_error < 0.001:
        print(f"\n✅ PASS: Int8 matmul is numerically accurate")
    else:
        print(f"\n⚠️  WARNING: Mean abs error {mean_abs_error:.6f} is high")

    return mse < 1e-5


if __name__ == "__main__":
    test_int8_matmul()
