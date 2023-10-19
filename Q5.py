# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from Q1_1 import generate_signal
from Q3 import get_neuron_response_to_current, plot_spikerate
from math import exp

def get_h_of_t(time, n, T):
    
    h_t = []
    
    c = np.sum(pow(time, n) * np.exp(-time/T))
    
    for t in time:
        
        if t < 0:
            h_t.append(0)
            
        else:
            h = pow(t, n) * exp(-t/T) / c
            h_t.append(h)      
            
    return h_t

dt = 0.001

time, current, frequencies, components = generate_signal(T = 2, dt = dt, power_desired = 0.5, limit = 5, seed = 12345)

voltages_pos_enc, spike_times_pos, pos_spikes_positions = get_neuron_response_to_current(time, dt, current, 1)
voltages_neg_enc, spike_times_neg, neg_spikes_positions = get_neuron_response_to_current(time, dt, current, -1)

#5A)
for n in [0, 1, 2]:
    h_t = get_h_of_t(time, n, 0.007)
    plt.plot(time[0:100], h_t[0:100], label="n = " + str(n))
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Scaling")
plt.title("5A) Postsynaptic Filters")
plt.ylim([0, 0.15])
plt.legend()
plt.show()

#5B)

discussion = """
    By increasing n, the filter is more spread out, and since its area is
    normalized to 1, it is shorter as well. This makes it a lower-pass filter
    the larger n is.
    
    Increasing n also pushes the filter further ahead in time, which
    means it will cause a delay to any function it convolves.
"""

#print(discussion)

#5C)
for T in [0.002, 0.005, 0.01, 0.02]:
    h_t = get_h_of_t(time, 0, T)
    plt.plot(time[0:50], h_t[0:50], label="Tau = " + str(T))
    
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Scaling")
plt.title("5C) Postsynaptic Filters")
plt.legend()
plt.show()

#5D)

discussion = """
    By increasing Tau, the filter is made more horizontal. Since the filter
    magnitude is normalized to 1, this means future values are given more
    and more weight, the greater Tau is, which increases lag in the system.
    
    This is similar behaviour to increasing n, except there is no emergent
    symmetric lump like when increasing n, just flatter and flatter curves
    that reach further and further into the future.
    
"""

#print(discussion)


#5E)

h = get_h_of_t(time, n=0, T=0.07)

r = np.array(pos_spikes_positions) - np.array(neg_spikes_positions)

R = np.fft.fft(r)

H = np.fft.fft(h)

XHAT = R * H

xhat = np.fft.ifft(XHAT)

plt.plot(time, xhat, label="xhat")
plt.plot(time, current, label="current")

plt.legend()
plt.title("5E")
plt.xlabel("y")
plt.ylabel("y")
plt.show()



