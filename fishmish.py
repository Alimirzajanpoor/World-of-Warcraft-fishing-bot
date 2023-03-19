import pyaudio
import numpy as np
import pyautogui
import time
from pynput.mouse import Button, Controller

# define the frequency range of the sound you want to detect
# adjust these values to match the frequency range of the sound you want to detect
freq_min = 2000
freq_max = 6000

# define the threshold value for detecting the sound
# adjust this value to fine-tune the detection algorithm`11`
threshold = 0.1

# define the interval between each right-click in seconds
click_interval = 1

# initialize the PyAudio object
p = pyaudio.PyAudio()

# open a stream to capture audio from the default input device
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True)

last_click_time = time.time()

while True:
    # read a chunk of audio data from the stream
    data = stream.read(1024)

    # convert the audio data to a numpy array
    samples = np.frombuffer(data, dtype=np.float32)

    # apply a Fast Fourier Transform (FFT) to the samples to get the frequency spectrum
    spectrum = np.fft.fft(samples)

    # compute the power spectrum by taking the square of the absolute values of the FFT coefficients
    power_spectrum = np.abs(spectrum) ** 2

    # get the frequency bins corresponding to the FFT coefficients
    freq_bins = np.fft.fftfreq(len(samples), 1.0 / 44100)

    # get the indices of the frequency bins that correspond to the desired frequency range
    freq_indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))[0]

    # compute the total power in the frequency range
    total_power = np.sum(power_spectrum[freq_indices])

    # if the total power is above the threshold and the time since the last click is greater than the click interval, simulate a right-click
    if total_power > threshold and time.time() - last_click_time >= click_interval:
        mouse = Controller()
        time.sleep(0.5)
        pyautogui.write('22')
        time.sleep(0.5)
        pyautogui.write('11')

        pyautogui.rightClick()
        last_click_time = time.time()
        # time.sleep(1)

        # pyautogui.write('22')
