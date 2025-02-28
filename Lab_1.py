import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter
import scipy.fft as fft
from scipy.signal.windows import blackman



# Filter tranfer function plotting
"""
def plot_filter(b, a, title, fs=1.0):
    w, h = freqz(b, a, worN=1024, fs=1)
    
    plt.figure(figsize=(12, 6))
    
    # Magnitude Response
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    
    # Phase Response
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h), 'g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid()
    
    # show 
    plt.tight_layout()
    plt.show()
    
b = [1,0.5] # Numerator coefficients
a = [1,-1] # Denominator coefficients

#plot_filter(b,a, "Custom Example IIR Filter")

b = [1,0.5, 0.25] # Numerator coefficients
a = [1] # Denominator coefficients

#plot_filter(b,a, "Custom Example FIR Filter")


b = [1,1,1,1,1] # Numerator coefficients
a = [1] # Denominator coefficients

#plot_filter(b,a, "Provided FIR Filter")

b = [1,1] # Numerator coefficients
a = [1,-1] # Denominator coefficients

#plot_filter(b,a, "Provided IIR Filter")
"""

# CT to DT conversions
'''
# Converts CT to DT signals: 
def get_signal(t): ## Primary frequency being sampled
    return np.sin(2 * np.pi * 300*10e6 * t)

def get_signal2(t): ## alias frequency 
    return np.sin(-2 * np.pi * 200*10e6 * t) 


fs = 500*10e6  # Sampling frequency
T = 1 / fs  # Sampling period
t_max = 1*10e-10 # time of coverage
t_cont = np.linspace(0, t_max, 1000)  # CT
t_disc = np.arange(0, t_max, T)  # DT

# Sample the signal
x_cont = get_signal(t_cont)
x_disc = get_signal(t_disc)
x_cont2 = get_signal2(t_cont)

plt.figure(figsize=(10, 5))
plt.plot(t_cont, x_cont, 'b', label='Continuous Signal')
plt.plot(t_cont, x_cont2, 'b', label='Continuous Signal (200MHz)', color = 'green')
plt.stem(t_disc, x_disc, 'r', linefmt='r-', markerfmt='ro', basefmt='r', label='Discrete Samples')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.title("Continuous to Discrete Signal Transformation")
plt.show()
'''

'''
# Signal Reconstruction
freq = 300*10e6
def get_signal(t): ## Primary frequency being sampled
    return np.sin(2 * np.pi * freq * t)


fs = 1000*10e6  # Sampling frequency
Ts = 1 / fs  # Sampling period
t_max1 = (10/freq)-Ts # time of coverage
t_min2 = Ts/2
t_max2 = (10/freq)-(Ts/2)
t_cont = np.linspace(0, t_max2, 1000)  # CT
t_reconstruct = np.linspace(0, t_max2, 1000)  # CT
#t_disc1 = np.arange(0, t_max1, Ts)  # DT 1 - on phase
t_disc1 = np.arange(t_min2, t_max2, Ts)  # DT 2 - out of phase

# Sample the signal
x_cont = get_signal(t_cont)
x_disc1 = get_signal(t_disc1)
#x_disc2 = get_signal(t_disc2)

# Reconstruction - sinc function used to simplify math (use normalized sinc function - no pi)
x_reconstruct = np.sum(x_disc1[:, np.newaxis] * np.sinc((t_reconstruct - t_disc1[:, np.newaxis]) / Ts), axis=0)

# Get MSE
MSE = np.mean((x_reconstruct - x_cont)**2)
print("MSE = ", MSE)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(t_cont, x_cont, 'b', label='Continuous Signal')
# Insert reconstructed signal
plt.stem(t_disc1, x_disc1, 'r', linefmt='r-', markerfmt='ro', basefmt='r', label='Discrete Samples')
plt.plot(t_reconstruct, x_reconstruct, 'g', label='Reconstructed Signal')
# second plot - plt.stem(t_disc2, x_disc2, 'g', linefmt='g-', markerfmt='go', basefmt='g', label='Discrete Samples 2')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.title("Reconstruction of a 300MHz signal sampled at 1000MHz")
plt.show()
'''


## FFT Plotting

# Blackman window
DFT_Len = 50 # Choose DFT Length
window = blackman(DFT_Len)
print(window)

freq = 200*10e6
freq2 = 400*10e6
def get_signal(t): ## Primary frequency being sampled
    signal = np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * freq2 * t)
    return signal * window


fs = 1*10e9  # Sampling frequency
Ts = 1 / fs  # Sampling period # time of coverage
t_max1 = Ts * DFT_Len # Chhose max time for the right size DFT
#t_cont = np.linspace(0, t_max1, 1000)  # CT
t_disc = np.arange(0, t_max1, Ts)  # DT

# Sample the signal
#x_cont = get_signal(t_cont)
x_disc = get_signal(t_disc)

# Compute fourier signals
fourier = fft.fft(x_disc)
frequencies = fft.fftfreq(x_disc.size, Ts)  # Frequency bins
frequencies = frequencies / 10
print(frequencies.size) # Check DFT Size


# Plot
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:frequencies.size//2], np.abs(fourier[:frequencies.size//2]))  # Only plot positive frequencies
plt.title("FFT of Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
