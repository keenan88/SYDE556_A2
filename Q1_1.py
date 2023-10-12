# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq



get_ipython().magic('clear')
get_ipython().magic('reset -f')



def generate_signal(T, dt, power_desired, limit, seed):
    
    np.random.seed(seed)
    
    N = int(T / dt)
    x = np.linspace(0, T, N, endpoint=False)
    
    X_w = np.zeros((N,), dtype=complex)
    X_w[0] = np.random.normal(0, 1) + 1j*np.random.normal(0, 1) # Set 0 frequency
    xf = fftfreq(N, dt) # Gives frequency at each index

    for freq_idx in range(1, len(X_w)//2):    
        if xf[freq_idx] < limit: # Only generate signals for frequencies that are below the band limit
            signal = np.random.normal(0, 1) + 1j*np.random.normal(0, 1)
            X_w[freq_idx] = signal
            X_w[-freq_idx] = np.conjugate(signal) # Set the negative frequency too, ifft needs the pos and neg frequency
            
    y = np.real(np.fft.ifft(X_w))

    power = np.sqrt(np.sum(np.square(y)) / T)

    scaling_factor = power_desired * np.sqrt(T / (np.sum(np.square(y))))

    y = y * scaling_factor
    X_w = X_w * scaling_factor

    power = np.sqrt(np.sum(np.square(y)) / T)

    print(power)
    
    return x, y, xf, X_w
    
    

"""
# FFT test
N = 600 # number of samples
T = 1.0 / 800.0 # sample spacing, each samples is 1/800th of a unit away from other samples (unit is seconds?)
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(5 * 2.0*np.pi*x) + 0.5*np.sin(8 * 2.0*np.pi*x) # Frequency is in rotations / second.
yf = fft(y) # Gives magnitude and offset of each frequency?
xf = fftfreq(N, T) # Gives frequency at each index of yf?
plt.plot(xf[:N//2], 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

# IFFT test

# Feed in the whole hog, positive and negative frequencies. 
# Gives back number of estimates equal to the number of frequencies fed in.
y_est = np.fft.ifft(yf) 


plt.plot(x, y_est, label="estimate")
plt.plot(x, y, label="original")
plt.legend()
plt.grid()
plt.show()
"""

#1A) 

T = 1
dt = 1 / 1000
power_desired = 0.5
seed = 18945

for limit in [5, 10, 20]:

    x, y, xf, X_w = generate_signal(T, dt, power_desired, limit, seed)

    plt.grid()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1A) Random signal limited at " + str(limit) + " Hz")
    plt.legend()
    plt.show()



#1B)















