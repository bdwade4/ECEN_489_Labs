#### NOTE: README
# This code was rewritten many times in order to fill the lab requirements
# The framework remains relatively unchanged throughout, but the current version will answer the final question

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter
import scipy.fft as fft
from scipy.signal.windows import blackman, hann, hamming
from scipy.optimize import minimize


################### Sampling ADC  ############################
    
# Sample the signal

# Converts CT to DT signals: 
DFT_Len = 100 # Choose DFT Length
variance = (0.5)**2
bits = 14
Vfs = 1

mismatch = 0.25 # Chosen SAR Capacitor mismatch

# Ideal quantization
def quantize(value):
    value = value * ((2**(bits-1))/Vfs) # Centered around 0 - multiply by half the bits
    #print(value)
    value = np.round(value) # each whole number is now a discrete point - round
    #print(value)
    value = value / (2**(bits-1))# Decimate back to true value of quantized signal
    return value

# Converts listed binary to conventional value
def binary_int(binary_list):
    binary_str = ''.join(str(bit) for bit in binary_list)
    decimal = int(binary_str, 2)
    return decimal

# Funciton adjusts threshold for mismatch
def adjust(threshold, opt, i):
    i = i + 1
    if (opt == 1):
        if (i in [2,6,8,9]):
            return (threshold * (1 + mismatch))
        elif (i in [11,14]):
            return (threshold * (1 - mismatch))
    elif (opt == 2):
        if (i in [3,12]):
            return (threshold * (1 + mismatch))
        elif (i in [6,9]):
            return (threshold * (1 - mismatch))
    else:
        return threshold
    return threshold

# Binary quantization (mimics SAR)
def binary_quantize(value,opt):
    value = (value + Vfs) * ((2**(bits-1))/(Vfs))  # Centered around 0 - multiply by half the bits
    
    
    value_list = [0] * (bits)
    for i in range(bits):
        x = abs(bits - i - 1)
        residue = value - binary_int(value_list) # Gets residue
        threshold = (2**x) 
        
        threshold = adjust(threshold, opt, i)
        
        # Append the corresponding binary
        if (residue >= threshold):
            value_list[i] = 1
        else:
            value_list[i] = 0
            
        '''
        print("Threshold: ", threshold)
        print("residue: ", residue)
        print("Binary: ", value_list)
        print("Current: ", binary_int(value_list))
        print()
        '''
    
    value = binary_int(value_list)
    value = (value / (2**(bits-1))) - Vfs # Decimate back to true value of quantized signal
    
    return value
        
    
noise_amp = 0.01
# OPT is the ADC chosen (0 = ideal, 1 = mismatch 1, 2 = mismatch 2)
def get_signal(t,opt): ## Primary frequency being sampled
    final = []
    for time in t:
        noise = (-noise_amp/2) + (noise_amp * 2 * np.random.rand())
        #noise = np.random.normal(0,np.sqrt(variance)) # Gaussian
        value = 0.2*(np.sin(2 * np.pi * 2*10e6 * time)) + noise
        # Round to quantized value (-1V to 1V Full scale range)
        value = binary_quantize(value,opt)
        # Append and return
        final.append(value)
    return final 


fs = 5.0001*10e6  # Sampling frequency
Ts = 1 / fs  # Sampling period
t_max = Ts * DFT_Len # time of coverage
t_cont = np.linspace(0, t_max, 1000)  # CT (approxmiation)
t_disc = np.arange(0, t_max, Ts)  # DT


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

################### Find the factors to minimize error (ML) #################
def ml_model(list_a, list_b, list_c): # Generates quadratic adjustments for each ADC
    a = np.array(list_a)
    b = np.array(list_b)
    c = np.array(list_c)

    # Fit x * a + b ≈ c
    def loss_a(params):
        scale, intercept = params
        return np.sum((scale * a + intercept - c) ** 2)

    # Fit y * b + c ≈ c
    def loss_b(params):
        scale, intercept = params
        return np.sum((scale * b + intercept - c) ** 2)

    res_a = minimize(loss_a, x0=[1, 0])
    res_b = minimize(loss_b, x0=[1, 0])

    x_scale, x_intercept = res_a.x
    y_scale, y_intercept = res_b.x
    
    a_coeffs = [x_scale,x_intercept]
    b_coeffs = [y_scale,y_intercept]

    return a_coeffs, b_coeffs


################### Combining logic for the two ADCs   ############################
def combine(x_ADC1, x_ADC2, plot): 
    x_out = [a + b for a, b in zip(x_ADC1, x_ADC2)]
    x_out = [x/2 for x in x_out]
    
    x_err = [a - b for a, b in zip(x_ADC1, x_ADC2)]
    
    if (plot == True):
        # Plot Error and Discrete Signal
        plt.figure(figsize=(10, 5))
        plt.plot(t_disc, x_out, 'b', label='Combined Signal')
        plt.stem(t_disc, x_err, 'r', linefmt='r-', markerfmt='ro', basefmt='r', label='Error')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.title("Plot of the Error and Combined Signal")
        plt.show()
    
    return x_out, x_err


################### Compute and Plot Fourier Analysis  ############################
def eval_fourier(x_out,plot): # Evaluates Fourier Transform
    # Compute fourier signals
    fourier = fft.fft(x_out)
    frequencies = fft.fftfreq(len(x_out), Ts)  # Frequency bins
    
    normalized_fourier = []
    mid = []
    fourier = (abs(fourier))**2 ###### GET THE PSD #####
    for freq in fourier:
        value = abs(freq)
        mid.append(value)
    
    second_largest = sorted(set(mid))[-2]
    # Normalize in dB to strongest signal 
    for val in mid:
        value = 10 * np.log10(val / abs(max(mid))) #for true normalizeation, den = abs(max(mid))
        normalized_fourier.append(value)
    
    # Calculate SNR
    SNR = np.average(normalized_fourier[:((frequencies.size//2) - 30)])
    #print("SNR: ", SNR)
    
    if (plot == True):
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies[:frequencies.size//2], normalized_fourier[:frequencies.size//2])  # Only plot positive frequencies
        plt.title("Normalized FFT of Signal")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Normalized Amplitude [dB]")
        plt.grid()
        plt.show()
    return SNR
        
        
################### Call Functions (Machine Learning)  ############################

def modify_from_ml(ADC1,ADC2):
    a1 = -0.044299053541603683 - 0.3275529758295806 + 0.11962223568095572 + 0.195108607639702
    a2 = 0.004874685132819728 + 0.001569162854879364 - 0.0016276687488099345 - 0.0011876487619025908
    b1 = -0.056391877729202185 - 0.257843997922653 + 0.11961568774286199 + 0.22409004144965178
    b2 = 0.004641050954413969 + 0.001234361316654111 - 0.0017446578640630478 - 0.0013945355480219448
    print()
    print("A1: ", a1)
    print("A2: ", a2)
    print("B1: ", b1)
    print("B2: ", b2)
    
    
    a = [a1 * x + a2 for x in ADC1]
    b = [b1 * x + b2 for x in ADC2]
    return a,b

# Returns a single SNR value from a trial
def run_cycle():
    x_ADC1 = get_signal(t_disc,1) # Use mismatch 1
    x_ADC2 = get_signal(t_disc,2) # Use mismatch 2
    
    # This enables/disables the modification of the function from previoys iterations of the ML model
    #x_ADC1,x_ADC2 = modify_from_ml(x_ADC1,x_ADC2) # Modify ML
    

    out, err = combine(x_ADC1,x_ADC2,False)
    
    x,y = ml_model(x_ADC1, x_ADC2, err) # Machine learning iterations
    
    '''
    print("X:", x)
    print("Y: ", y)
    print()
    '''
    
    snr = eval_fourier(out, False)
    return snr,x,y
    
total = []
x_total = []
y_total = []
for i in range(100):
    snr,x,y = run_cycle()
    total.append(snr)
    x_total.append(x)
    y_total.append(y)
    
# Average factors of correction
x_total = np.array(x_total)
x = x_total.T
y_total = np.array(y_total)
y = y_total.T

print()
print(np.average(total))
print("ADC1: ",np.average(x[0]),"X +",np.average(x[1]))
print("ADC2: ",np.average(y[0]),"Y +",np.average(y[1]))






