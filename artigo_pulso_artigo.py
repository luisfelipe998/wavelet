import numpy as np
import matplotlib.pyplot as plt
import pywt

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

# --- Autocorrelation and Cross-correlation ---
def unbiased_autocorrelation(x):
    N = len(x)
    return np.array([
        np.sum(x[:N - m] * np.conj(x[m:])) / (N - m)
        for m in range(N)
    ])

def unbiased_crosscorrelation(x, y):
    N = len(x)
    return np.array([
        np.sum(x[:N - m] * np.conj(y[m:])) / (N - m)
        for m in range(N)
    ])

Rxx = unbiased_autocorrelation(signal1)
Ryy = unbiased_autocorrelation(signal2)
Rxy = unbiased_crosscorrelation(signal1, signal2)

# --- Wavelet Transform of Correlations (Alternative Definitions) ---
scales = np.arange(1, 128)
wavelet = 'cmor0.5-1.5'
WPS1_alt, freqs = pywt.cwt(Rxx.real, scales, wavelet, sampling_period=1 / sampling_rate)
WPS2_alt, _ = pywt.cwt(Ryy.real, scales, wavelet, sampling_period=1 / sampling_rate)
WCS_alt, _ = pywt.cwt(Rxy.real, scales, wavelet, sampling_period=1 / sampling_rate)

# --- Coherence (Alternative WCO Definition) ---
WCO_alt = np.abs(WCS_alt) ** 2 / (np.abs(WPS1_alt[:, [0]]) * np.abs(WPS2_alt[:, [0]]))

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
plt.title("Alternative WPS (Original Signal) from $R_{xx}(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(np.abs(WPS2_alt), extent=[0, duration, scales[-1], scales[0]], cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.title("Alternative WPS (Delayed Signal) from $R_{yy}(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(np.abs(WCS_alt), extent=[0, duration, scales[-1], scales[0]], cmap='coolwarm', aspect='auto')
plt.colorbar(label='Cross Power')
plt.title("Alternative Wavelet Cross Spectrum from $R_{xy}(t)$")
plt.ylabel("Scale")
plt.xlabel("Time [s]")

plt.figure(figsize=(12, 4))
plt.imshow(WCO_alt, extent=[0, duration, scales[-1], scales[0]], cmap='plasma', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Coherence')
plt.title("Alternative Wavelet Coherence")
plt.ylabel("Scale")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()
