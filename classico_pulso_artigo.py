import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import gaussian_filter

# --- Signal Parameters ---
duration = 2.0
sampling_rate = 1000  # Hz
num_samples = int(duration * sampling_rate)
t = np.linspace(0, duration, num_samples, endpoint=False)

# --- Modulated Gaussian Pulse ---
pulse_length = 100
t_pulse = np.linspace(-1, 1, pulse_length)
modulated_gaussian = np.exp(-t_pulse**2) * np.cos(6 * t_pulse)

# --- White Noise ---
np.random.seed(123)
noise_std = 0.05
white_noise1 = np.random.normal(0, noise_std, num_samples)
white_noise2 = np.random.normal(0, noise_std, num_samples)

# --- Compose signals ---
signal1 = white_noise1.copy()
signal2 = white_noise2.copy()
start_index = 200
signal1[start_index:start_index + pulse_length] += modulated_gaussian
delay_samples = 400
signal2[start_index + delay_samples:start_index + delay_samples + pulse_length] += modulated_gaussian

# --- Wavelet Transform ---
scales = np.arange(1, 128)
wavelet = 'cmor0.5-1.5'
coef1, _ = pywt.cwt(signal1, scales, wavelet, sampling_period=1 / sampling_rate)
coef2, _ = pywt.cwt(signal2, scales, wavelet, sampling_period=1 / sampling_rate)

# Power Spectrum and Cross Spectrum
WPS1_alt = np.abs(coef1) ** 2
WPS2_alt = np.abs(coef2) ** 2
WCS_alt = coef1 * np.conj(coef2)

# Gaussian smoothing
def smooth_gaussian(x, sigma=(2, 2)):
    return gaussian_filter(np.abs(x), sigma=sigma)

sWCS = smooth_gaussian(WCS_alt)
sWPS1 = smooth_gaussian(WPS1_alt)
sWPS2 = smooth_gaussian(WPS2_alt)

# Coherence
WCO_alt = np.abs(sWCS) ** 2 / (sWPS1 * sWPS2)

# --- Plotting ---
plt.figure(figsize=(10, 4))
plt.plot(t, signal1, label='Original Signal')
plt.plot(t, signal2, label='Delayed Signal', alpha=0.7)
plt.title("Original and Delayed Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.imshow(np.abs(WPS1_alt), extent=[0, duration, scales[-1], scales[0]], cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.title("Classical WPS (Original Signal) from $x(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(np.abs(WPS2_alt), extent=[0, duration, scales[-1], scales[0]], cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.title("Classical WPS (Delayed Signal) from $y(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(np.abs(WCS_alt), extent=[0, duration, scales[-1], scales[0]], cmap='coolwarm', aspect='auto')
plt.colorbar(label='Cross Power')
plt.title("Classical Wavelet Cross Spectrum from $x(t)$ and $y(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(WCO_alt, extent=[0, duration, scales[-1], scales[0]], cmap='plasma', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Coherence')
plt.title("Classical Wavelet Coherence (Gaussian Smoothing)")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()
