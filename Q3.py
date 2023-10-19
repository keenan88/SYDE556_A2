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
    spike_times = []
    spikes_positions = []
    i = 0

    while i < len(time):
        J = alpha * encoder * current[int(i)] + J_bias
        v += dt * (J - v) / Trc
        
        if v < 0: #Normalize any negative voltage to 0.
            v = 0

        if v >= Vth:
            v = 0
            spike_times.append(time[int(i - 2)])
            spikes_positions.append(encoder)
            spikes_positions.append(0)
            
            ref_ms = Tref * 100
            j = 0
            while j < ref_ms:
                voltages.append(0)
                j += 1
                i += 1
            
        else: # No spike
            voltages.append(v)
            spikes_positions.append(0)
            i += 1
            
    return voltages[0:len(time)], spike_times[0:len(time)], spikes_positions[0:len(time)]
    
def plot_spikerate(time, spike_times, current, voltage, title):
    plt_lim = -1
    plt.grid()
    plt.plot(time[0:plt_lim], voltage[0:plt_lim])
    plt.plot(time[0:plt_lim], current[0:plt_lim])
    
    for spike_time in spike_times:
        if spike_time < time[plt_lim]:
            continue
            plt.axvline(x = spike_time, color = "black")
        
    
    plt.ylabel("Voltage")
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
    
    voltages_pos_enc, spike_times = get_neuron_response_to_current(time, dt, current, 1)    
    plot_spikerate(time, spike_times, current, voltages_pos_enc, "3A) Voltage spikes against time, encoder = 1, constant current of 0")
        
    voltages_neg_enc, spike_times = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_times, current, voltages_neg_enc, "3A) Voltage spikes against time, encoder = -1, constant current of 0")
    
    current += 1
    voltages_pos_enc, spike_times = get_neuron_response_to_current(time, dt, current, 1)        
    plot_spikerate(time, spike_times, current, voltages_pos_enc, "3A) Voltage spikes against time, encoder = 1, constant current of 1")
     
    voltages_neg_enc, spike_times = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_times, current, voltages_neg_enc, "3A) Voltage spikes against time, encoder = -1, constant current of 1")
    
    
    # 3B)
    
    current = 0.5 * np.sin(10 * np.pi * time)
    
    voltages, spike_times = get_neuron_response_to_current(time, dt, current, 1)
    plot_spikerate(time, spike_times, current, voltages, "3B) Voltage spikes against time, encoder = 1, current = 0.5 * Sin(10 * pi * t)")
    
    voltages, spike_times = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_times, current, voltages, "3B) Voltage spikes against time, encoder = -1, current = 0.5 * Sin(10 * pi * t)")
    
    
    
    # 3C)
    
    time, current, _, _ = generate_signal(T = 2, dt = 0.001, power_desired = 0.5, limit = 5, seed = 12345)
    
    voltages, spike_times = get_neuron_response_to_current(time, dt, current, 1)
    plot_spikerate(time, spike_times, current, voltages, "3C) Current and spike output for 5Hz bandlimited random current, encoder = 1")
    
    voltages, spike_times = get_neuron_response_to_current(time, dt, current, -1)
    plot_spikerate(time, spike_times, current, voltages, "3C) Current and spike output for 5Hz bandlimited random current, encoder = -1")
