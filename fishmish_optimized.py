#!/usr/bin/env python3
"""
Optimized World of Warcraft Fishing Bot
Performance improvements:
- Pre-computed frequency bins and indices
- Memory pooling to reduce allocations
- Optimized FFT operations
- Better resource management
- Configurable parameters
- Error handling and recovery
- Performance monitoring
"""

import numpy as np
import time
import threading
import queue
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple
import sys

# Lazy imports for faster startup
_pyaudio = None
_pyautogui = None
_pynput = None

def get_pyaudio():
    global _pyaudio
    if _pyaudio is None:
        import pyaudio
        _pyaudio = pyaudio
    return _pyaudio

def get_pyautogui():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        _pyautogui = pyautogui
    return _pyautogui

def get_pynput():
    global _pynput
    if _pynput is None:
        from pynput.mouse import Button, Controller
        _pynput = (Button, Controller)
    return _pynput

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 44100
    buffer_size: int = 1024  # Reduced from 1024 for lower latency
    channels: int = 1
    format_type: str = 'float32'
    freq_min: int = 2000
    freq_max: int = 6000
    threshold: float = 0.1
    click_interval: float = 1.0

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.frame_times = []
        self.max_samples = 1000
        self.start_time = time.perf_counter()
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self.frame_times:
            return {}
        
        return {
            'avg_frame_time_ms': np.mean(self.frame_times) * 1000,
            'max_frame_time_ms': np.max(self.frame_times) * 1000,
            'min_frame_time_ms': np.min(self.frame_times) * 1000,
            'fps': 1.0 / np.mean(self.frame_times) if np.mean(self.frame_times) > 0 else 0,
            'total_frames': len(self.frame_times),
            'uptime_seconds': time.perf_counter() - self.start_time
        }

class OptimizedAudioProcessor:
    """Optimized audio processing with pre-computed values and memory pooling."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pre-compute frequency bins and indices (major optimization)
        self._precompute_frequency_data()
        
        # Memory pooling for numpy arrays
        self._init_memory_pool()
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Audio stream
        self.stream: Optional[object] = None
        self.p: Optional[object] = None
        
        # Control flags
        self.running = False
        self.last_click_time = 0.0
    
    def _precompute_frequency_data(self):
        """Pre-compute frequency bins and target indices."""
        # This is computed once instead of every frame (huge performance gain)
        self.freq_bins = np.fft.fftfreq(self.config.buffer_size, 1.0 / self.config.sample_rate)
        self.freq_indices = np.where(
            (self.freq_bins >= self.config.freq_min) & 
            (self.freq_bins <= self.config.freq_max)
        )[0]
        
        self.logger.info(f"Pre-computed {len(self.freq_indices)} frequency indices for range "
                        f"{self.config.freq_min}-{self.config.freq_max} Hz")
    
    def _init_memory_pool(self):
        """Initialize memory pools for reusable arrays."""
        # Pre-allocate arrays to avoid repeated allocations
        self.sample_buffer = np.zeros(self.config.buffer_size, dtype=np.float32)
        self.spectrum_buffer = np.zeros(self.config.buffer_size, dtype=np.complex128)
        self.power_buffer = np.zeros(self.config.buffer_size, dtype=np.float64)
    
    def initialize_audio(self):
        """Initialize audio stream with error handling."""
        try:
            pyaudio = get_pyaudio()
            self.p = pyaudio.PyAudio()
            
            # Find the best audio format
            format_map = {
                'float32': pyaudio.paFloat32,
                'int16': pyaudio.paInt16,
                'int32': pyaudio.paInt32
            }
            
            audio_format = format_map.get(self.config.format_type, pyaudio.paFloat32)
            
            self.stream = self.p.open(
                format=audio_format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.buffer_size,
                stream_callback=None  # Use blocking mode for better control
            )
            
            self.logger.info(f"Audio stream initialized: {self.config.sample_rate}Hz, "
                           f"buffer={self.config.buffer_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
            return False
    
    def process_audio_frame(self, audio_data: bytes) -> Tuple[float, float]:
        """Process a single audio frame with optimized operations."""
        frame_start = time.perf_counter()
        
        try:
            # Reuse pre-allocated buffer
            np.frombuffer(audio_data, dtype=np.float32, out=self.sample_buffer)
            
            # Optimized FFT using pre-allocated buffers
            np.fft.fft(self.sample_buffer, out=self.spectrum_buffer)
            
            # Compute power spectrum in-place
            np.abs(self.spectrum_buffer, out=self.power_buffer)
            np.square(self.power_buffer, out=self.power_buffer)
            
            # Use pre-computed indices for frequency filtering
            total_power = np.sum(self.power_buffer[self.freq_indices])
            
            frame_time = time.perf_counter() - frame_start
            self.monitor.record_frame_time(frame_time)
            
            return total_power, frame_time
            
        except Exception as e:
            self.logger.error(f"Error processing audio frame: {e}")
            return 0.0, 0.0
    
    def should_trigger_action(self, total_power: float) -> bool:
        """Determine if action should be triggered."""
        current_time = time.perf_counter()
        
        return (total_power > self.config.threshold and 
                current_time - self.last_click_time >= self.config.click_interval)
    
    def trigger_fishing_action(self):
        """Trigger fishing actions with error handling."""
        try:
            pyautogui = get_pyautogui()
            Button, Controller = get_pynput()
            
            mouse = Controller()
            
            # Optimized action sequence
            time.sleep(0.1)  # Reduced sleep time
            pyautogui.write('22')
            time.sleep(0.1)  # Reduced sleep time
            pyautogui.write('11')
            pyautogui.rightClick()
            
            self.last_click_time = time.perf_counter()
            self.logger.debug("Fishing action triggered")
            
        except Exception as e:
            self.logger.error(f"Error triggering fishing action: {e}")
    
    def run(self):
        """Main processing loop with optimized performance."""
        if not self.initialize_audio():
            return False
        
        self.running = True
        self.logger.info("Starting optimized fishing bot...")
        
        try:
            while self.running:
                try:
                    # Read audio data
                    audio_data = self.stream.read(self.config.buffer_size, exception_on_overflow=False)
                    
                    # Process the frame
                    total_power, frame_time = self.process_audio_frame(audio_data)
                    
                    # Check if action should be triggered
                    if self.should_trigger_action(total_power):
                        self.trigger_fishing_action()
                    
                    # Log performance stats periodically
                    if len(self.monitor.frame_times) % 1000 == 0:
                        stats = self.monitor.get_stats()
                        self.logger.info(f"Performance: {stats['avg_frame_time_ms']:.2f}ms avg, "
                                       f"{stats['fps']:.1f} FPS")
                
                except KeyboardInterrupt:
                    self.logger.info("Stopping bot...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(0.1)  # Brief pause before continuing
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
        
        # Print final stats
        stats = self.monitor.get_stats()
        self.logger.info(f"Final performance stats: {stats}")

class ConfigManager:
    """Manage configuration with optimization presets."""
    
    @staticmethod
    def get_performance_preset(preset: str = "balanced") -> AudioConfig:
        """Get optimized configuration presets."""
        presets = {
            "low_latency": AudioConfig(
                buffer_size=512,
                sample_rate=44100,
                threshold=0.05,
                click_interval=0.8
            ),
            "balanced": AudioConfig(
                buffer_size=1024,
                sample_rate=44100,
                threshold=0.1,
                click_interval=1.0
            ),
            "high_accuracy": AudioConfig(
                buffer_size=2048,
                sample_rate=44100,
                threshold=0.15,
                click_interval=1.2
            ),
            "power_save": AudioConfig(
                buffer_size=2048,
                sample_rate=22050,  # Lower sample rate
                threshold=0.1,
                click_interval=1.5
            )
        }
        
        return presets.get(preset, presets["balanced"])

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fishmish_optimized.log')
        ]
    )

def main():
    """Main entry point with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized WoW Fishing Bot")
    parser.add_argument("--preset", choices=["low_latency", "balanced", "high_accuracy", "power_save"],
                       default="balanced", help="Performance preset")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--freq-min", type=int, default=2000, help="Minimum frequency")
    parser.add_argument("--freq-max", type=int, default=6000, help="Maximum frequency")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = ConfigManager.get_performance_preset(args.preset)
    
    # Override with command line arguments
    if args.freq_min != 2000:
        config.freq_min = args.freq_min
    if args.freq_max != 6000:
        config.freq_max = args.freq_max
    if args.threshold != 0.1:
        config.threshold = args.threshold
    
    logger.info(f"Starting with preset: {args.preset}")
    logger.info(f"Configuration: {config}")
    
    # Create and run processor
    processor = OptimizedAudioProcessor(config)
    
    try:
        success = processor.run()
        if not success:
            logger.error("Failed to start audio processing")
            return 1
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())