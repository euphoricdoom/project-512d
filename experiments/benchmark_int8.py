"""
Benchmark int8 vs float32 encoder loading.
Measures: load time, memory footprint, disk I/O.

Shows the real-world system advantages of int8 quantization.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def get_process_memory():
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        print("Install psutil for memory measurement: pip install psutil")
        return None


def benchmark_load(model_path: str, format_name: str, num_runs: int = 5):
    """Benchmark loading a model file."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {format_name}")
    print(f"{'='*70}")

    file_size = Path(model_path).stat().st_size / 1024 / 1024
    print(f"  File size: {file_size:.2f} MB")

    io_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with open(model_path, 'rb') as f:
            _ = f.read()
        io_times.append(time.perf_counter() - start)

    avg_io = float(np.mean(io_times))
    std_io = float(np.std(io_times))
    print(f"  Disk I/O:  {avg_io*1000:.2f} +/- {std_io*1000:.2f} ms")

    load_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        data = np.load(model_path)
        if "encoder_W_int8" in data:
            _ = data["encoder_W_int8"]
            _ = data["encoder_scale"]
        elif "encoder_W" in data:
            _ = data["encoder_W"]
        data.close()
        load_times.append(time.perf_counter() - start)

    avg_load = float(np.mean(load_times))
    std_load = float(np.std(load_times))
    print(f"  NPZ load:  {avg_load*1000:.2f} +/- {std_load*1000:.2f} ms")

    mem_before = get_process_memory()
    data = np.load(model_path)
    if "encoder_W_int8" in data:
        W_int8 = data["encoder_W_int8"]
        scale = float(data["encoder_scale"])
        W = W_int8.astype(np.float32) * scale
        format_detail = "int8 -> float32"
    else:
        W = data["encoder_W"]
        format_detail = "float32"

    mem_after = get_process_memory()
    data.close()

    mem_delta = None
    if mem_before is not None and mem_after is not None:
        mem_delta = mem_after - mem_before
        print(f"  Memory:    {mem_delta:.2f} MB ({format_detail})")
    else:
        print(f"  Memory:    N/A (install psutil)")

    print(f"  Encoder shape: {W.shape}")
    print(f"  Encoder dtype: {W.dtype}")

    return {
        'file_size_mb': file_size,
        'io_time_ms': avg_io * 1000,
        'load_time_ms': avg_load * 1000,
        'memory_mb': mem_delta,
        'total_time_ms': (avg_io + avg_load) * 1000,
    }


def create_float32_reference(model_path: str, output_path: str):
    """Create a float32 version of an int8 model for comparison."""
    print(f"\nCreating float32 reference from {model_path}...")
    data = np.load(model_path)

    if "encoder_W_int8" in data:
        W_int8 = data["encoder_W_int8"]
        scale = float(data["encoder_scale"])
        W_float32 = W_int8.astype(np.float32) * scale
    else:
        print("  Model is already float32, using as-is")
        return model_path

    arrays = {"encoder_W": W_float32}
    for key in data.files:
        if key.startswith("head_"):
            arrays[key] = data[key]

    np.savez_compressed(output_path, **arrays)
    data.close()
    print(f"  Saved -> {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


def main():
    """Run comprehensive int8 vs float32 benchmark."""
    print("="*70)
    print("INT8 vs FLOAT32 ENCODER BENCHMARK")
    print("="*70)

    int8_model = str(ROOT / "test_int8.npz")
    if not Path(int8_model).exists():
        int8_model = str(ROOT / "quality_int8.npz")
    if not Path(int8_model).exists():
        print("No int8 model found. Run training first:")
        print("  python train_hf.py --dataset imdb --task test --epochs 100 --reservoir-dim 1536 --model-out test_int8.npz")
        sys.exit(1)

    float32_ref = str(ROOT / "temp_float32_ref.npz")
    float32_model = create_float32_reference(int8_model, float32_ref)

    results_float32 = benchmark_load(float32_model, "FLOAT32")
    results_int8    = benchmark_load(int8_model, "INT8")

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    size_ratio   = results_float32['file_size_mb'] / results_int8['file_size_mb']
    io_speedup   = results_float32['io_time_ms'] / results_int8['io_time_ms']
    load_speedup = results_float32['load_time_ms'] / results_int8['load_time_ms']
    total_speedup = results_float32['total_time_ms'] / results_int8['total_time_ms']

    print(f"  File size:  {results_float32['file_size_mb']:.2f} MB -> {results_int8['file_size_mb']:.2f} MB ({size_ratio:.1f}x smaller)")
    print(f"  Disk I/O:   {results_float32['io_time_ms']:.2f} ms -> {results_int8['io_time_ms']:.2f} ms ({io_speedup:.2f}x faster)")
    print(f"  NPZ load:   {results_float32['load_time_ms']:.2f} ms -> {results_int8['load_time_ms']:.2f} ms ({load_speedup:.2f}x faster)")
    print(f"  Total time: {results_float32['total_time_ms']:.2f} ms -> {results_int8['total_time_ms']:.2f} ms ({total_speedup:.2f}x faster)")

    if results_float32['memory_mb'] is not None and results_int8['memory_mb'] is not None:
        print(f"  Runtime RAM: {results_float32['memory_mb']:.2f} MB -> {results_int8['memory_mb']:.2f} MB")

    if Path(float32_ref).exists():
        Path(float32_ref).unlink()


if __name__ == "__main__":
    main()
