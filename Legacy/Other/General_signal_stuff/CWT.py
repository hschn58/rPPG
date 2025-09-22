import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate synthetic earthquake-like signal
np.random.seed(42)  # For reproducibility
sampling_rate = 100  # samples per second
t = np.arange(0, 10, 1/sampling_rate)  # 10-second time vector

# Simulate an earthquake signal: combine a few sine waves with different frequencies and add noise
frequencies = [1, 5, 10, 20]  # frequencies in Hz
signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies) + 0.5 * np.random.randn(len(t))

# Compute Continuous Wavelet Transform using the Morlet wavelet
scales = np.arange(1, 128)  # range of scales to use in the CWT
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/sampling_rate)

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Synthetic Earthquake Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the CWT scalogram
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 10, 1, 128], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=abs(coefficients).min())
plt.yscale('log')
plt.title('CWT Scalogram (Morlet Wavelet)')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
