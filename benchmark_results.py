#!/usr/bin/env python3
"""
Benchmark runner that demonstrates the performance improvements.
This script runs comprehensive benchmarks and saves results.
"""

import time
import numpy as np
import json
import sys
from datetime import datetime
import platform

def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    
    print("Running Performance Benchmark Suite")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        },
        'benchmarks': {}
    }
    
    # FFT Performance Benchmark
    print("1. FFT Performance Benchmark")
    fft_results = benchmark_fft_performance()
    results['benchmarks']['fft_performance'] = fft_results
    
    # Frequency Filtering Benchmark
    print("\n2. Frequency Filtering Benchmark")
    filter_results = benchmark_frequency_filtering()
    results['benchmarks']['frequency_filtering'] = filter_results
    
    # Memory Usage Benchmark
    print("\n3. Memory Usage Benchmark")
    memory_results = benchmark_memory_usage()
    results['benchmarks']['memory_usage'] = memory_results
    
    # Overall Performance Benchmark
    print("\n4. Overall Performance Benchmark")
    overall_results = benchmark_overall_performance()
    results['benchmarks']['overall_performance'] = overall_results
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to benchmark_results.json")
    
    return results

def benchmark_fft_performance():
    """Benchmark FFT operations with different approaches."""
    print("  Testing FFT operations...")
    
    buffer_sizes = [512, 1024, 2048]
    iterations = 1000
    results = {}
    
    for buffer_size in buffer_sizes:
        print(f"    Buffer size: {buffer_size}")
        
        # Generate test data
        samples = np.random.random(buffer_size).astype(np.float32)
        
        # Standard approach
        start_time = time.perf_counter()
        for _ in range(iterations):
            spectrum = np.fft.fft(samples)
            power = np.abs(spectrum) ** 2
        standard_time = time.perf_counter() - start_time
        
        # Optimized approach
        spectrum_buffer = np.zeros(buffer_size, dtype=np.complex128)
        power_buffer = np.zeros(buffer_size, dtype=np.float64)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            np.fft.fft(samples, out=spectrum_buffer)
            np.abs(spectrum_buffer, out=power_buffer)
            np.square(power_buffer, out=power_buffer)
        optimized_time = time.perf_counter() - start_time
        
        speedup = standard_time / optimized_time
        
        results[f'buffer_{buffer_size}'] = {
            'standard_time_ms': standard_time * 1000,
            'optimized_time_ms': optimized_time * 1000,
            'speedup': speedup,
            'improvement_percent': ((standard_time - optimized_time) / standard_time) * 100
        }
        
        print(f"      Standard: {standard_time * 1000:.2f}ms")
        print(f"      Optimized: {optimized_time * 1000:.2f}ms")
        print(f"      Speedup: {speedup:.2f}x")
    
    return results

def benchmark_frequency_filtering():
    """Benchmark frequency filtering approaches."""
    print("  Testing frequency filtering...")
    
    buffer_size = 1024
    freq_min, freq_max = 2000, 6000
    iterations = 5000
    
    # Generate test data
    samples = np.random.random(buffer_size).astype(np.float32)
    spectrum = np.fft.fft(samples)
    power_spectrum = np.abs(spectrum) ** 2
    
    # Original approach (compute indices every time)
    start_time = time.perf_counter()
    for _ in range(iterations):
        freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
        freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
        total_power = np.sum(power_spectrum[freq_indices])
    original_time = time.perf_counter() - start_time
    
    # Optimized approach (pre-computed indices)
    freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
    freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        total_power = np.sum(power_spectrum[freq_indices])
    optimized_time = time.perf_counter() - start_time
    
    speedup = original_time / optimized_time
    
    results = {
        'original_time_ms': original_time * 1000,
        'optimized_time_ms': optimized_time * 1000,
        'speedup': speedup,
        'improvement_percent': ((original_time - optimized_time) / original_time) * 100
    }
    
    print(f"    Original: {original_time * 1000:.2f}ms")
    print(f"    Optimized: {optimized_time * 1000:.2f}ms")
    print(f"    Speedup: {speedup:.2f}x")
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("  Testing memory usage...")
    
    import tracemalloc
    
    def original_processing(num_frames):
        for _ in range(num_frames):
            data = np.random.random(1024).astype(np.float32).tobytes()
            samples = np.frombuffer(data, dtype=np.float32)
            spectrum = np.fft.fft(samples)
            power_spectrum = np.abs(spectrum) ** 2
            freq_bins = np.fft.fftfreq(len(samples), 1.0 / 44100)
    
    def optimized_processing(num_frames):
        buffer_size = 1024
        sample_buffer = np.zeros(buffer_size, dtype=np.float32)
        spectrum_buffer = np.zeros(buffer_size, dtype=np.complex128)
        power_buffer = np.zeros(buffer_size, dtype=np.float64)
        
        for _ in range(num_frames):
            data = np.random.random(buffer_size).astype(np.float32).tobytes()
            np.frombuffer(data, dtype=np.float32, out=sample_buffer)
            np.fft.fft(sample_buffer, out=spectrum_buffer)
            np.abs(spectrum_buffer, out=power_buffer)
    
    num_frames = 100
    
    # Test original approach
    tracemalloc.start()
    original_processing(num_frames)
    current_orig, peak_orig = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Test optimized approach
    tracemalloc.start()
    optimized_processing(num_frames)
    current_opt, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    results = {
        'original_peak_mb': peak_orig / 1024 / 1024,
        'optimized_peak_mb': peak_opt / 1024 / 1024,
        'memory_reduction_percent': ((peak_orig - peak_opt) / peak_orig) * 100,
        'original_current_mb': current_orig / 1024 / 1024,
        'optimized_current_mb': current_opt / 1024 / 1024
    }
    
    print(f"    Original peak: {peak_orig / 1024 / 1024:.2f} MB")
    print(f"    Optimized peak: {peak_opt / 1024 / 1024:.2f} MB")
    print(f"    Memory reduction: {results['memory_reduction_percent']:.1f}%")
    
    return results

def benchmark_overall_performance():
    """Benchmark overall application performance."""
    print("  Testing overall performance...")
    
    def simulate_original(num_frames):
        freq_min, freq_max = 2000, 6000
        threshold = 0.1
        
        total_time = 0
        for _ in range(num_frames):
            start = time.perf_counter()
            
            # Simulate audio processing
            data = np.random.random(1024).astype(np.float32).tobytes()
            samples = np.frombuffer(data, dtype=np.float32)
            spectrum = np.fft.fft(samples)
            power_spectrum = np.abs(spectrum) ** 2
            freq_bins = np.fft.fftfreq(len(samples), 1.0 / 44100)
            freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
            total_power = np.sum(power_spectrum[freq_indices])
            
            if total_power > threshold:
                pass
            
            total_time += time.perf_counter() - start
        
        return total_time
    
    def simulate_optimized(num_frames):
        freq_min, freq_max = 2000, 6000
        threshold = 0.1
        buffer_size = 1024
        
        # Pre-compute frequency data
        freq_bins = np.fft.fftfreq(buffer_size, 1.0 / 44100)
        freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]
        
        # Pre-allocate buffers
        sample_buffer = np.zeros(buffer_size, dtype=np.float32)
        spectrum_buffer = np.zeros(buffer_size, dtype=np.complex128)
        power_buffer = np.zeros(buffer_size, dtype=np.float64)
        
        total_time = 0
        for _ in range(num_frames):
            start = time.perf_counter()
            
            # Simulate audio processing
            data = np.random.random(buffer_size).astype(np.float32).tobytes()
            np.frombuffer(data, dtype=np.float32, out=sample_buffer)
            np.fft.fft(sample_buffer, out=spectrum_buffer)
            np.abs(spectrum_buffer, out=power_buffer)
            np.square(power_buffer, out=power_buffer)
            total_power = np.sum(power_buffer[freq_indices])
            
            if total_power > threshold:
                pass
            
            total_time += time.perf_counter() - start
        
        return total_time
    
    num_frames = 1000
    
    print(f"    Processing {num_frames} frames...")
    
    # Benchmark original
    original_time = simulate_original(num_frames)
    original_fps = num_frames / original_time
    
    # Benchmark optimized
    optimized_time = simulate_optimized(num_frames)
    optimized_fps = num_frames / optimized_time
    
    speedup = original_time / optimized_time
    
    results = {
        'frames_processed': num_frames,
        'original_total_time_ms': original_time * 1000,
        'original_avg_frame_time_ms': (original_time / num_frames) * 1000,
        'original_fps': original_fps,
        'optimized_total_time_ms': optimized_time * 1000,
        'optimized_avg_frame_time_ms': (optimized_time / num_frames) * 1000,
        'optimized_fps': optimized_fps,
        'speedup': speedup,
        'performance_improvement_percent': ((original_time - optimized_time) / original_time) * 100
    }
    
    print(f"    Original: {original_time * 1000:.2f}ms total, {original_fps:.1f} FPS")
    print(f"    Optimized: {optimized_time * 1000:.2f}ms total, {optimized_fps:.1f} FPS")
    print(f"    Speedup: {speedup:.2f}x")
    
    return results

def print_summary(results):
    """Print benchmark summary."""
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    # Overall performance
    overall = results['benchmarks']['overall_performance']
    print(f"Overall Performance Improvement: {overall['speedup']:.2f}x faster")
    print(f"Frame Processing: {overall['optimized_avg_frame_time_ms']:.2f}ms avg (was {overall['original_avg_frame_time_ms']:.2f}ms)")
    print(f"Effective FPS: {overall['optimized_fps']:.1f} (was {overall['original_fps']:.1f})")
    
    # Memory usage
    memory = results['benchmarks']['memory_usage']
    print(f"Memory Usage Reduction: {memory['memory_reduction_percent']:.1f}%")
    print(f"Peak Memory: {memory['optimized_peak_mb']:.2f}MB (was {memory['original_peak_mb']:.2f}MB)")
    
    # Frequency filtering
    filtering = results['benchmarks']['frequency_filtering']
    print(f"Frequency Filtering Speedup: {filtering['speedup']:.2f}x")
    
    print("\nKey Optimizations:")
    print("✓ Pre-computed frequency bins and indices")
    print("✓ Memory pooling with reusable buffers")
    print("✓ In-place numpy operations")
    print("✓ Optimized FFT operations")
    print("✓ Reduced memory allocations")

if __name__ == "__main__":
    try:
        results = run_benchmark_suite()
        print_summary(results)
        print(f"\nDetailed results saved to benchmark_results.json")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install numpy to run benchmarks")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)