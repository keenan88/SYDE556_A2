# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from math import exp



get_ipython().magic('clear')
get_ipython().magic('reset -f')



def generate_signal_band_reduced(T, dt, power_desired, bw_hz, seed):
    
    np.random.seed(seed)
    
    bw_rads = bw_hz * np.pi * 2
    
    N = int(T / dt)
    x = np.linspace(0, T, N, endpoint=False)
    
    X_w = np.zeros((N,), dtype=complex)
    freq = 0
    stdev = exp((-freq * freq) / (2 * bw_rads * bw_rads))
    X_w[0] = np.random.normal(0, stdev) + 1j*np.random.normal(0, stdev) # Set 0 frequency
    xf_rads = fftfreq(N, dt) * 2 * np.pi # Gives frequency at each index

    for freq_idx in range(1, len(X_w)//2):    
        freq = xf_rads[freq_idx]
        stdev = exp((-freq * freq) / (2 * bw_rads * bw_rads))
        signal = np.random.normal(0, stdev) + 1j*np.random.normal(0, stdev)
        X_w[freq_idx] = signal
        X_w[-freq_idx] = np.conjugate(signal) # Set the negative frequency too, ifft needs the pos and neg frequency
            
    y = np.real(np.fft.ifft(X_w))

    scaling_factor = power_desired / np.sqrt(np.sum(y**2) / N)

    y = y * scaling_factor
    X_w = X_w * scaling_factor
    
    power = np.sqrt(np.mean(y**2))
    
    print("Power: ", power)
    
    return x, y, xf_rads, X_w


#1.2A) 

T = 1
dt = 1 / 1000
N = int(T/dt)
power_desired = 0.5
seed = 18945
bw = 5

for bw_hz in [5, 10, 20]:

    x, y, xf_rads, X_w = generate_signal_band_reduced(T, dt, power_desired, bw_hz, seed)

    plt.grid()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1.2A) Random signal bandwidth reduced at " + str(bw) + " Hz")
    plt.legend()
    plt.show()

    
#1.2B)
bw_hz = 20

summed_Xws = np.zeros(N, dtype=complex)

for i in range(100):
    
    x, y, xf_rads, X_w = generate_signal_band_reduced(T, dt, power_desired, bw_hz, seed + i)
    
    summed_Xws += np.abs(X_w)
    
mean_Xws = summed_Xws / 100

plt.scatter(xf_rads[:N//2][0:100], 2.0/N * np.abs(mean_Xws[0:N//2])[0:100])
plt.grid()
plt.xlabel("w in radians")
plt.ylabel("mag(X(w))")
plt.title("1.1B) Average power spectrum of random signal bandlimited at 10 Hz")
plt.show()










