#### NOTE: README
# This code was rewritten many times in order to fill the lab requirements
# The framework 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter
import scipy.fft as fft
from scipy.signal.windows import blackman, hann, hamming


################### Part 2:  ############################
    
# Sample the signal

# Converts CT to DT signals: 
DFT_Len = 100 # Choose DFT Length
variance = 50e-4
bits = 12
Vfs = 1
def quantize(value):
    value = value * ((2**(bits-1))/Vfs) # Centered around 0 - multiply by half the bits
    #print(value)
    value = np.round(value) # each whole number is now a discrete point - round
    #print(value)
    value = value / (2**(bits-1))# Decimate back to true value of quantized signal
    return value
    
def get_signal(t): ## Primary frequency being sampled
    final = []
    for time in t:
        noise = np.random.normal(0,np.sqrt(variance))
        value = (np.sin(2 * np.pi * 200*10e6 * time) + noise)
        # Round to quantized value (-1V to 1V Full scale range)
        value = quantize(value)
        # Append and return
        final.append(value)
    return final * hann(DFT_Len)

DFT_Len = 100 # Choose DFT Length


fs = 641.16129032258*10e6  # Sampling frequency
Ts = 1 / fs  # Sampling period
t_max = Ts * DFT_Len # time of coverage
t_cont = np.linspace(0, t_max, 1000)  # CT (approxmiation)
t_disc = np.arange(0, t_max, Ts)  # DT

# Sample the signal
#x_cont = get_signal(t_cont)
x_disc = get_signal(t_disc)

'''
# Plot the sampled DT/CT signal 
plt.figure(figsize=(10, 5))
plt.plot(t_cont, x_cont, 'b', label='Continuous Signal')
plt.stem(t_disc, x_disc, 'r', linefmt='r-', markerfmt='ro', basefmt='r', label='Discrete Samples')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.title("Continuous to Discrete Signal Transformation")
plt.show()
'''

# Compute fourier signals
fourier = fft.fft(x_disc)
frequencies = fft.fftfreq(len(x_disc), Ts)  # Frequency bins

normalized_fourier = []
mid = []
fourier = (abs(fourier))**2 ###### GET THE PSD #####
for freq in fourier:
    value = abs(freq)
    mid.append(value)

# Normalize in dB to strongest signal 
for val in mid:
    value = 10 * np.log10(val / abs(max(mid)))
    normalized_fourier.append(value)

# Calculate SNR
SNR = np.average(normalized_fourier[:((frequencies.size//2) - 30)])
print("SNR: ", SNR)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:frequencies.size//2], normalized_fourier[:frequencies.size//2])  # Only plot positive frequencies
plt.title("Normalized FFT of Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized Amplitude [dB]")
plt.grid()
plt.show()