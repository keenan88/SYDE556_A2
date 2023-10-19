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

dt = 0.001

time, current, frequencies, components = generate_signal(T = 2, dt = dt, power_desired = 5, limit = 5, seed = 12345)



voltages_pos_enc, spike_times_pos, pos_spikes_positions = get_neuron_response_to_current(time, dt, current, 1)
voltages_neg_enc, spike_times_neg, neg_spikes_positions = get_neuron_response_to_current(time, dt, current, -1)

#plot_spikerate(time, spike_times, current, voltages_pos_enc, "4) Random current input, positive encoder")
#plot_spikerate(time, spike_times, current, voltages_pos_enc, "4) Random current input, negative encoder")

x = current
X = components
spikes = np.array([
    pos_spikes_positions, 
    neg_spikes_positions
])

plt.plot(time, current)

plt.plot(time, pos_spikes_positions)
plt.plot(time, neg_spikes_positions)

plt.show()

# x and X should (effectively) be 1D-arrays
assert x.ndim == 1 and X.ndim == 1
assert x.shape[0] == X.shape[0]

# Nt will represent the number of samples present in the current and spikes signal.
Nt = x.size

# Make sure that "spikes" is a 2 x Nt array
assert spikes.ndim == 2
assert spikes.shape[0] == 2              
assert spikes.shape[1] == Nt

# Optimal filter for 2 neurons (Page 9/18 on slides 4) is defined as: 
# H(w) = X(w) * norm(R(w)) / Mag(R(w))^2.
# H(w) is convolved with the spike trains, in an effort to make the spike trains
# Better resemble the original input.

# X(w) is the frequency components of the input signal.
# R(w) is the frequency components of d * (a1 - a2), the decoded signal.
# R(w) * H(w) should give a signal that best approximates X(w)

# Time length of signal is number of samples multiplied by time length of each sample
T = Nt * dt

# Time series centered about t = 0, spanning number of samples, with step size dt.
ts = np.arange(Nt) * dt - T / 2.0


# Difference between spikerate of pos encoded neuron and neg encoded neuron
r = spikes[0] - spikes[1]

# fourier transform of the difference between the pos encoded and neg encoded neuron's voltages.
R = np.fft.fftshift(np.fft.fft(r))

# Frequency series equal to number of samples, normalized against T.
# Subtracting Nt / (2 * T) centers the series about 0, I expect to deal with fft
# providing pos and neg frequencies. Cut frequencies down to half the number of samples
# To reach the nyquist frequency and no further.
fs = np.arange(Nt) / T - Nt / (2.0 * T)

# Tunable parameter for the window function W2.
# Increasing sigma_t will cause the window to be tighter
# Decreasing sigma_t will cause the window to be wider.
sigma_t = 25e-3

# Converting frequencies (in arbitrary Hz, cycles/second) to rotations per second.
omega = fs * 2.0 * np.pi

# Setup of window function to tune H. W2 is based on omega, which is based on fs, 
# and fs is centered about 0, so this system is not causaul, and the window function
# W2 will be symmetrically smoothing all datapoints in a convolution.
# Since omega is linear and centered about 0, the window function will be
# A gaussian centered about 0 as well.
W2 = np.exp(-omega**2*sigma_t**2)

# By normalizing W2 to 1, convolving with W2 will not scale the convolved function,
# Just change its shape
W2 = W2 / sum(W2)

# This is the numerator of H, determined with the error minimization equation,
# without convolution with the window function W2.
# Note that since X was generated with specific signals, X(w) has a few very strong
# Frequencies, and all other signals are effectively 0.
CP = X * R.conjugate()

# This is the numerator of H, including convolution with W2. So now the numerator
# Has been smoothed by W2, the window function. This means that all frequencies
# Will get atleast some representation, instead of just the handful of strong
# Frequencies present in CP.
WCP = np.convolve(CP, W2, 'same')

# The denominator of H, without any smoothing. Since R is being multiplied by its
# conjugate, strong frequencies will be scaled up much more than weak frequencies,
# So some smoothing could be useful to generate a signal more realistic to nature.
RP = R * R.conjugate()

# The denominator of H, smoothed.
WRP = np.convolve(RP, W2, 'same')

# Magnitudes squared of the components of random current signal. Not used in H.
XP = X * X.conjugate()

# Convolution of random signal components and transformed frequency scale. Not used in H.
# Will be used to compare the decoded filtered smoothed signal against.
WXP = np.convolve(XP, W2, 'same')

# Final step in creating filter H.
H = WCP / WRP

# Taking our filter out of fourier domain and back into real domain
h = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H))).real

# Convolve H with R, so that at each spike in R, an H is "spawned", best approximating
# The original x stimulus.
XHAT = H*R

# This is just taking the difference in voltages back out of the fourier domain into the time domain
xhat = np.fft.ifft(np.fft.ifftshift(XHAT)).real

plt.scatter(time, xhat, color='blue')
plt.plot(time, current, color='orange')
plt.grid()

plt.show()




print("Made it!")


