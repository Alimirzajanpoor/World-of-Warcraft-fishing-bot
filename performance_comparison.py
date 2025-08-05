#!/usr/bin/env python3
"""
Performance comparison between original and optimized fishmish implementations.
This script benchmarks both versions without requiring actual audio hardware.
"""

import time
import numpy as np
import cProfile
import pstats
import io
from contextlib import contextmanager
import sys

@contextmanager
def timer(description):
    """Context manager to time code execution."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start) * 1000:.2f}ms")

def simulate_original_processing(num_frames=1000):
    """Simulate the original fishmish.py processing logic."""
    freq_min = 2000
    freq_max = 6000
    threshold = 0.1
    
    total_time = 0
    for _ in range(num_frames):
        start = time.perf_counter()
        
        # Simulate reading audio data
        data = np.random.random(1024).astype(np.float32).tobytes()
        samples = np.frombuffer(data, dtype=np.float32)
        
        # Original processing (inefficient)
        spectrum = np.fft.fft(samples)
        power_spectrum = np.abs(spectrum) ** 2
        freq_bins = np.fft.fftfreq(len(samples), 1.0 / 44100)  # Computed every frame!
        freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
        total_power = np.sum(power_spectrum[freq_indices])
        
        if total_power > threshold:
            pass  # Would trigger actions
        
        total_time += time.perf_counter() - start
    
    return total_time

def simulate_optimized_processing(num_frames=1000):
    """Simulate the optimized processing logic."""
    freq_min = 2000
    freq_max = 6000
    threshold = 0.1
    buffer_size = 1024
    
    # Pre-compute frequency data (optimization #1)
    freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
    freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
    
    # Pre-allocate buffers (optimization #2)
    sample_buffer = np.zeros(buffer_size, dtype=np.float32)
    spectrum_buffer = np.zeros(buffer_size, dtype=np.complex128)
    power_buffer = np.zeros(buffer_size, dtype=np.float64)
    
    total_time = 0
    for _ in range(num_frames):
        start = time.perf_counter()
        
        # Simulate reading audio data
        data = np.random.random(buffer_size).astype(np.float32).tobytes()
        
        # Optimized processing
        np.frombuffer(data, dtype=np.float32, out=sample_buffer)
        np.fft.fft(sample_buffer, out=spectrum_buffer)
        np.abs(spectrum_buffer, out=power_buffer)
        np.square(power_buffer, out=power_buffer)
        total_power = np.sum(power_buffer[freq_indices])  # Use pre-computed indices
        
        if total_power > threshold:
            pass  # Would trigger actions
        
        total_time += time.perf_counter() - start
    
    return total_time

def benchmark_fft_operations():
    """Benchmark different FFT approaches."""
    print("=== FFT Operation Benchmarks ===")
    
    buffer_sizes = [512, 1024, 2048, 4096]
    iterations = 1000
    
    for buffer_size in buffer_sizes:
        print(f"\nBuffer size: {buffer_size}")
        
        # Test data
        samples = np.random.random(buffer_size).astype(np.float32)
        
        # Standard approach (new allocation each time)
        with timer("  Standard FFT"):
            for _ in range(iterations):
                spectrum = np.fft.fft(samples)
                power = np.abs(spectrum) ** 2
        
        # Optimized approach (pre-allocated buffers)
        spectrum_buffer = np.zeros(buffer_size, dtype=np.complex128)
        power_buffer = np.zeros(buffer_size, dtype=np.float64)
        
        with timer("  Optimized FFT"):
            for _ in range(iterations):
                np.fft.fft(samples, out=spectrum_buffer)
                np.abs(spectrum_buffer, out=power_buffer)
                np.square(power_buffer, out=power_buffer)

def benchmark_frequency_filtering():
    """Benchmark frequency filtering approaches."""
    print("\n=== Frequency Filtering Benchmarks ===")
    
    buffer_size = 1024
    freq_min, freq_max = 2000, 6000
    iterations = 10000
    
    samples = np.random.random(buffer_size).astype(np.float32)
    spectrum = np.fft.fft(samples)
    power_spectrum = np.abs(spectrum) ** 2
    
    # Original approach (compute indices every time)
    with timer("Original filtering"):
        for _ in range(iterations):
            freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
            freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
            total_power = np.sum(power_spectrum[freq_indices])
    
    # Optimized approach (pre-computed indices)
    freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
    freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
    
    with timer("Optimized filtering"):
        for _ in range(iterations):
            total_power = np.sum(power_spectrum[freq_indices])

def memory_usage_comparison():
    """Compare memory usage patterns."""
    print("\n=== Memory Usage Comparison ===")
    
    import tracemalloc
    
    # Original approach
    tracemalloc.start()
    simulate_original_processing(100)
    current_orig, peak_orig = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Optimized approach
    tracemalloc.start()
    simulate_optimized_processing(100)
    current_opt, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Original - Current: {current_orig / 1024 / 1024:.2f} MB, Peak: {peak_orig / 1024 / 1024:.2f} MB")
    print(f"Optimized - Current: {current_opt / 1024 / 1024:.2f} MB, Peak: {peak_opt / 1024 / 1024:.2f} MB")
    print(f"Memory reduction: {((peak_orig - peak_opt) / peak_orig * 100):.1f}%")

def full_performance_comparison():
    """Compare overall performance between original and optimized versions."""
    print("\n=== Full Performance Comparison ===")
    
    num_frames = 1000
    
    print(f"Processing {num_frames} audio frames...")
    
    # Benchmark original
    original_time = simulate_original_processing(num_frames)
    original_fps = num_frames / original_time
    
    # Benchmark optimized
    optimized_time = simulate_optimized_processing(num_frames)
    optimized_fps = num_frames / optimized_time
    
    # Results
    print(f"\nOriginal version:")
    print(f"  Total time: {original_time * 1000:.2f}ms")
    print(f"  Average per frame: {(original_time / num_frames) * 1000:.2f}ms")
    print(f"  Effective FPS: {original_fps:.1f}")
    
    print(f"\nOptimized version:")
    print(f"  Total time: {optimized_time * 1000:.2f}ms")
    print(f"  Average per frame: {(optimized_time / num_frames) * 1000:.2f}ms")
    print(f"  Effective FPS: {optimized_fps:.1f}")
    
    speedup = original_time / optimized_time
    print(f"\nPerformance improvement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time reduction: {((original_time - optimized_time) / original_time * 100):.1f}%")

def profile_comparison():
    """Profile both implementations to identify bottlenecks."""
    print("\n=== Profiling Comparison ===")
    
    # Profile original
    print("\nOriginal implementation profile:")
    pr = cProfile.Profile()
    pr.enable()
    simulate_original_processing(100)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())
    
    # Profile optimized
    print("\nOptimized implementation profile:")
    pr = cProfile.Profile()
    pr.enable()
    simulate_optimized_processing(100)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())

def main():
    """Run all performance comparisons."""
    print("Performance Comparison: Original vs Optimized Fishmish")
    print("=" * 60)
    
    try:
        benchmark_fft_operations()
        benchmark_frequency_filtering()
        memory_usage_comparison()
        full_performance_comparison()
        profile_comparison()
        
        print("\n" + "=" * 60)
        print("Performance analysis complete!")
        print("\nKey optimizations implemented:")
        print("1. Pre-computed frequency bins and indices")
        print("2. Memory pooling with pre-allocated buffers")
        print("3. In-place numpy operations")
        print("4. Lazy imports for faster startup")
        print("5. Better error handling and resource management")
        print("6. Configurable performance presets")
        print("7. Built-in performance monitoring")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install numpy: pip install numpy")
        return 1
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())