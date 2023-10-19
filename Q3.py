# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from Q1_1 import generate_signal
from math import exp

def get_neuron_response_to_current(time, dt, current, encoder):
    Trc = 20/1000
    Tref = 2/1000
    
    x0 = 0
    a0 = 40
    x1 = 1
    a1 = 150
    
    J_bias = 1 / (1 - exp( (Tref - 1 / a0 ) / Trc))
    alpha = (1 / (1 - exp( (Tref - 1 / a1 ) / Trc)) - J_bias) / np.dot(1, x1)

    Vth = 1
    v = 0
    voltages = []
    spike_train = []
    i = 0

    while i < len(time):
        J = alpha * encoder * current[int(i)] + J_bias
        v += dt * (J - v) / Trc
        
        if v < 0: #Normalize any negative voltage to 0.
            v = 0

        if v >= Vth:
            v = 0
            spike_train.append(encoder)
            spike_train.append(0)
            
            ref_ms = Tref * 100
            j = 0
            while j < ref_ms:
                voltages.append(0)
                j += 1
                i += 1
            
        else: # No spike
            voltages.append(v)
            spike_train.append(0)
            i += 1
            
    return voltages[0:len(time)], spike_train[0:len(time)]
    
def plot_spikerate(time, spike_train, current, plt_lim, title):
    plt.grid()
    plt.plot(time[0:plt_lim], current[0:plt_lim])
    plt.plot(time[0:plt_lim], spike_train[0:plt_lim])
        
    
    plt.ylabel("Spiketrain & Stimulus")
    plt.xlabel("Time")
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    
    dt = 1 / 1000
    T = 1
    N = int(T / dt)
    current = np.zeros(N)
    
    time = np.linspace(0, T, N)
    
    # 3A)
    
    voltages_pos_enc, spike_train = get_neuron_response_to_current(time, dt, current, 1)    
    plot_spikerate(time, spike_train, current, -1, "3A) Spiketrain vs Stimulus, Encoder = 1, Current = 0")
        
    voltages_neg_enc, spike_train = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_train, current, -1, "3A) Spiketrain vs Stimulus, Encoder = -1, Current = 0")
    
    current += 1
    voltages_pos_enc, spike_train = get_neuron_response_to_current(time, dt, current, 1)        
    plot_spikerate(time, spike_train, current, -1, "3A) Spiketrain vs Stimulus, Encoder = 1, Current = 1")
     
    voltages_neg_enc, spike_train = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_train, current, -1, "3A) Spiketrain vs Stimulus, Encoder = -1, Current = 1")
    
    
    # 3B)
    
    current = 0.5 * np.sin(10 * np.pi * time)
    
    voltages, spike_train = get_neuron_response_to_current(time, dt, current, 1)
    plot_spikerate(time, spike_train, current, -1, "3B) Voltage spikes against time, encoder = 1, current = 0.5 * Sin(10 * pi * t)")
    
    voltages, spike_train = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_train, current, -1, "3B) Voltage spikes against time, encoder = -1, current = 0.5 * Sin(10 * pi * t)")
    
    
    
    # 3C)
    
    time, current, _, _ = generate_signal(T = 2, dt = 0.001, power_desired = 0.5, limit = 5, seed = 12345)
    
    voltages, spike_train = get_neuron_response_to_current(time, dt, current, 1)
    plot_spikerate(time, spike_train, current, -1, "3C) Current and spike output for 5Hz bandlimited random current, encoder = 1")
    
    voltages, spike_train = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_train, current, -1, "3C) Current and spike output for 5Hz bandlimited random current, encoder = -1")
