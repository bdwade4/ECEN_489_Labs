#### NOTE: README
# This code was rewritten many times in order to fill the lab requirements
# The framework remains relatively unchanged throughout, but the current version will answer the final question

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter
import scipy.fft as fft
from scipy.signal.windows import blackman, hann, hamming

# Import ML Libraries
from sklearn.ensemble import RandomForestRegressor


################### Sampling ADC  ############################
    
# Sample the signal

# Converts CT to DT signals: 
DFT_Len = 10000 # Choose DFT Length
variance = (0.5)**2
bits = 14
Vfs = 1

mismatch = 0.5 # Chosen SAR Capacitor mismatch

# Ideal quantization
def quantize(value):
    value = value * ((2**(bits-1))/Vfs) # Centered around 0 - multiply by half the bits
    #print(value)
    value = np.round(value) # each whole number is now a discrete point - round
    #print(value)
    value = value / (2**(bits-1))# Decimate back to true value of quantized signal
    return value

# Converts listed binary to conventional value (ChatGPT Assisted)
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
        value = 1*(np.sin(2 * np.pi * 0.7*10e6 * time)) + noise
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
fourier_len = 150 # lets me train ML model while not graphing all data
def eval_fourier(x_out,plot,choose): # Evaluates Fourier Transform
    # Compute fourier signals
    x_out = x_out[:fourier_len]
    fourier = fft.fft(x_out)
    frequencies = fft.fftfreq(len(x_out), Ts)  # Frequency bins
    
    normalized_fourier = []
    mid = []
    fourier = (abs(fourier))**2 ###### GET THE PSD #####
    for freq in fourier:
        value = abs(freq)
        mid.append(value)
    
    if (choose == 0):
        choice = max(mid)
    else: 
        choice = sorted(set(mid))[-2]
    
    # Normalize in dB to strongest signal 
    for val in mid:
        value = 10 * np.log10(val / abs(choice)) #for true normalizeation, den = abs(max(mid))
        normalized_fourier.append(value)
    
    # Calculate SNR
    SNR = np.average(normalized_fourier[((frequencies.size//2)):])
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

#Note: This is the only part of this project that primarily uses ChatGPT
def ML_Forest(ADC1,ADC2,err,out):
    # Intiialize the inputs 
    list1 = list(ADC1)
    list2 = list(ADC2)
    error = list(err)
    goal = list(out)
    
    # Manual Trimming of high error values
    i = 0
    for x in range(len(error)): 
        if (error[i] > (0.02*Vfs)):
            del error[i]
            del list1[i]
            del list2[i]
            del goal[i]
            i = i - 1 
        i = i + 1




    # Stack inputs
    x = np.column_stack((list1, list2))  # Shape (N, 2)

    # set the target output as the sum of the two - this should reduce the error
        # The interpolation should reduce the errors
    y = goal

    # Define the Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=500,        # Number of trees
        max_depth=None,          # Let trees grow until all leaves are pure
        random_state=250          # For reproducibility
    )

    # Optional: sample weights based on error
    sample_weights = [(1 / (20*x + 1)) for x in error]  # Inverse: smaller error = higher importance

    # Train the model
    model.fit(x, y, sample_weight=sample_weights)

    # Predict
    stacked = np.column_stack((ADC1, ADC2))
    predicted_output = model.predict(stacked)

    return predicted_output
    

############################# Final Combinations and Untilization ###############################

# Returns a single SNR value from a trial
def run_cycle():
    # Set mismatch to 0 for no error
    x_ADC1 = get_signal(t_disc,1) # Use mismatch 1
    x_ADC2 = get_signal(t_disc,2) # Use mismatch 2
    
    out, err = combine(x_ADC1,x_ADC2,False)
    
    # calls the ML function and returns the revised output from the random forest 
    revised = ML_Forest(x_ADC1,x_ADC2,err,out)
    
    snr = eval_fourier(out, False, 0) #1 is used if it is suspected that the DC value may be higher than the signal
    snr_new = eval_fourier(revised, False, 0)
    return snr,snr_new
    
total_snr = []
total_snr_new = []
for i in range(35):
    snr,snr_new = run_cycle()
    total_snr.append(snr)
    if (snr_new > -150): #Cuts out infinite values (error)
        total_snr_new.append(snr_new)

print()
print("Old SNR: ", np.average(total_snr))
print("Revised SNR: ", np.average(total_snr_new))




