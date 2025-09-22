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
coefficients, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1/sampling_rate)

# Convert scales to frequencies using the wavelet function properties
frequencies = pywt.scale2frequency('morl', scales) / (1/sampling_rate)

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title('Synthetic Earthquake Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the CWT scalogram with frequencies on y-axis
plt.subplot(3, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 10, frequencies.min(), frequencies.max()],
           cmap='jet', aspect='auto', vmax=abs(coefficients).max(), vmin=abs(coefficients).min())
plt.yscale('log')  # Use logarithmic scale for frequency axis
plt.title('CWT Scalogram (Morlet Wavelet)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')
plt.tight_layout()


plt.subplot(3, 1, 3)

# Function to perform the inverse CWT
def inverse_cwt(coefficients, scales, wavelet_name='morl', sampling_period=1.0):
    # Reconstruct the signal by summing over scales
    reconstructed_signal = np.zeros(coefficients.shape[1])
    
    for i, scale in enumerate(scales):
        # Generate the scaled wavelet for the current scale
        wavelet_function = pywt.ContinuousWavelet(wavelet_name)
        wavelet, _ = wavelet_function.wavefun()
        
        # Rescale the wavelet to match the scale used in the CWT
        scaled_wavelet = wavelet / np.sqrt(scale)
        scaled_wavelet = np.interp(np.arange(0, len(scaled_wavelet)) * scale, np.arange(0, len(scaled_wavelet)), scaled_wavelet)
        
        # Add the contribution of the current scale to the reconstructed signal
        for j in range(coefficients.shape[1]):
            reconstructed_signal[j] += coefficients[i, j] * scaled_wavelet[(j % len(scaled_wavelet))]
    
    # Normalize the reconstructed signal
    reconstructed_signal /= np.sqrt(scales[:, np.newaxis]).sum()
    
    return reconstructed_signal

# Reconstruct the signal from the CWT coefficients
reconstructed_signal = inverse_cwt(coefficients, scales, 'morl', sampling_period=1/sampling_rate)

# Plot the original and reconstructed signals for comparison

plt.plot(t, signal, label='Original Signal')
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='--')
plt.title('Original vs Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.show()
