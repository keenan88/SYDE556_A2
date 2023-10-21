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

get_ipython().magic('clear')
get_ipython().magic('reset -f')

dt = 0.001

time, stimulus, frequencies, components = generate_signal(T = 2, dt = dt, power_desired = 0.5, limit_hz = 5, seed = 12345)

voltages_pos_enc, pos_spiketrains = get_neuron_response_to_current(time, dt, stimulus, 1)
voltages_neg_enc, neg_spiketrains = get_neuron_response_to_current(time, dt, stimulus, -1)
neg_spiketrains = -1 * np.array(neg_spiketrains)

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

# "and using that as your activity matrix A to compute your decode" what??
# I thought the filtered spikes were attempting to be the stimulus.
# Like we don't have an A matrix of spikerates compared to stimulii..
# This seems wack.

h = get_h_of_t(time, n=0, T=0.007)



r = np.array(pos_spiketrains) - np.array(neg_spiketrains)

xhat = np.zeros(len(r))

for i in range(len(r)):
    
    if r[i]:
        
        xhat[i:] += r[i] * np.array(h[0: len(xhat) - i])

plt.plot(time, r, color='blue', label='differential spiketrain')        
plt.plot(time, xhat, color='red', label="x_hat")
plt.plot(time, stimulus, color='black', label="stimulus")
plt.legend()
plt.grid()
plt.title("5E)")
plt.show()

for i in range(len(xhat)):
    if xhat[i] == 0:
        xhat[i] = 0.001 # To avoid a singular matrix

A = np.matrix(xhat)

decoders = np.linalg.inv(A.T * A) * A.T * np.matrix(stimulus)

# A.T * A is singular, and that's no surprise since the first few
# Values of A are zero...

#stimulus_hat = np.array(decoders * A.T)


plt.legend()
plt.title("5E")
plt.xlabel("y")
plt.ylabel("y")
plt.show()



