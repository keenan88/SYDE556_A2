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



def generate_signal(T, dt, power_desired, bw, seed):
    
    np.random.seed(seed)
    
    N = int(T / dt)
    x = np.linspace(0, T, N, endpoint=False)
    
    X_w = np.zeros((N,), dtype=complex)
    X_w[0] = np.random.normal(0, 1) + 1j*np.random.normal(0, 1) # Set 0 frequency
    xf = fftfreq(N, dt) # Gives frequency at each index

    for freq_idx in range(1, len(X_w)//2):    
        freq = xf[freq_idx]
        stdev = exp((-freq * freq) / (2 * bw * bw))
        signal = np.random.normal(0, stdev) + 1j*np.random.normal(0, stdev)
        X_w[freq_idx] = signal
        X_w[-freq_idx] = np.conjugate(signal) # Set the negative frequency too, ifft needs the pos and neg frequency
            
    y = np.real(np.fft.ifft(X_w))

    scaling_factor = power_desired * np.sqrt(T / (np.sum(np.square(y))))

    y = y * scaling_factor
    X_w = X_w * scaling_factor
    
    return x, y, xf, X_w
    
    



#1.2A) 

T = 1
dt = 1 / 1000
N = int(T/dt)
power_desired = 0.5
seed = 18945
bw = 5

for bw in [5, 10, 20]:

    x, y, xf, X_w = generate_signal(T, dt, power_desired, bw, seed)

    plt.grid()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1.2A) Random signal bandwidth reduced at " + str(bw) + " Hz")
    plt.legend()
    plt.show()

    
#1.2B)
bw = 20

summed_Xws = np.zeros(N, dtype=complex)

for i in range(100):
    
    x, y, xf, X_w = generate_signal(T, dt, power_desired, bw, seed + i)
    
    summed_Xws += X_w
    
mean_Xws = summed_Xws / 100

xf = fftfreq(N, dt) # Gives frequency at each index of yf?
plt.plot(xf[:N//2][0:100], 2.0/N * np.abs(mean_Xws[0:N//2])[0:100])
plt.grid()
plt.xlabel("w in radians")
plt.ylabel("mag(X(w))")
plt.title("1.1B) Average power spectrum of random signal bandlimited at 10 Hz")
plt.show()










