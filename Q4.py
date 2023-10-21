# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from Q1_1 import generate_signal
from Q3 import get_neuron_response_to_current

get_ipython().magic('clear')
get_ipython().magic('reset -f')

def compute_optimal_filter(
        # Signal generated from your white noise generator
        x,
        # Fourier coefficients from your white noise generator
        X,
        # Spike train from the previous part
        spikes,
        # Time step size
        dt=1e-3
    ):

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
    
    
    # Difference between spikerate of pos encoded neuron and neg encoded neuron.
    # Given that the negative spiketrain spikes down, this is effectively normalizing
    # Negative spikes to positive spikes and combining both spiketrains into
    # One summed spiketrain.
    r = spikes[0] - spikes[1]
    
    # Frequency representation of summed spiketrain. Probably has an interesting
    # Distribution since r is basically a bunch of impulse signals.
    R = np.fft.fftshift(np.fft.fft(r))
    
    # Setting up range of window function W2
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
    
    return ts, fs, R, H, h, XHAT, xhat, XP, WXP


dt = 0.001

time, current, frequencies, components = generate_signal(T = 2, dt = dt, power_desired = 0.5, limit_hz = 5, seed = 1225)

voltages_pos_enc, pos_spiketrain = get_neuron_response_to_current(time, dt, current, 1)
voltages_neg_enc, neg_spiketrain = get_neuron_response_to_current(time, dt, current, -1)


x = current
X = components
X = np.roll(X, int(len(X)/2)) # This is necessary so that the frequencies of F line up with the frequencies of ts
frequencies = np.roll(frequencies, int(len(frequencies)/2)) # Just useful for comparing the frequencies of X_w to the frequencies expected by ts

spikes = np.array([
    pos_spiketrain, 
    -1 * np.array(neg_spiketrain)
])


ts, fs, R, H, h, XHAT, xhat, XP, WXP = compute_optimal_filter(x, X, spikes, dt=1e-3)

#B)
B_plt_lim = 800
plt.scatter(fs[B_plt_lim:-B_plt_lim], abs(H)[B_plt_lim:-B_plt_lim])
plt.title("4B) Magnitude of H(w) for each frequency")
plt.xlabel("Frequency (hz)")
plt.ylabel("Magnitude(H(w))")
plt.show()

plt.scatter(ts[B_plt_lim:-B_plt_lim], h[B_plt_lim:-B_plt_lim])
plt.title("4B) Filter H in time domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Strength")
plt.show()

#C)
plt.plot(time, pos_spiketrain, color='blue', label="Positive Neuron Spiketrain")
plt.plot(time, neg_spiketrain, color='orange', label="Negative Neuron Spiketrain")
plt.plot(time, xhat, color='red', label="Xhat")
plt.plot(time, current, color='black', label="True x")
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Voltage & Current")
plt.title("4C) x(t), x_hat(t), and neuron spikes")

plt.show()

#D)

D_plot_lim = 950
plt.plot(fs[D_plot_lim:-D_plot_lim], abs(X)[D_plot_lim:-D_plot_lim])
plt.title("4D) Magnitude Response of X(w)")
plt.xlabel("Frequency (hz)")
plt.ylabel("Magnitude(H(w))")
plt.grid()
plt.show()

plt.plot(fs, abs(R))
plt.title("4D) Magnitude Response of R(w)")
plt.xlabel("Frequency (hz)")
plt.ylabel("Magnitude(H(w))")
plt.grid()
plt.show()

plt.plot(fs[D_plot_lim:-D_plot_lim], abs(XHAT)[D_plot_lim:-D_plot_lim])
plt.title("4D) Magnitude Response of X_hat(w)")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude(H(w))")
plt.grid()
plt.show()

#4E)

discussion = """
How do these spectra relate to the optimal filter?

R(w) is the frequency representation spiking output of the filters. R(w) looks
nothing like X(w), the true stimulus. Once filtering is applied to R(w),
the resultant X_hat(W) looks much more like X(w).

The most noteworthy difference between X(w) and X_hat(w) is that X_hat(w) does
not have the same 5hz cutoff, so it has some lower-powered frequencies extending
past the cutoff point, and X_hat(w)'s lower frequencies dont have quite the same magnitude
that X(w)'s lower frequencies do.

"""

print(discussion)

#4F)

for limit_hz_ in [2, 10, 30]:
    
    time, current, frequencies, components = generate_signal(T = 2, dt = dt, power_desired = 0.5, limit_hz = limit_hz_, seed = 1225)

    voltages_pos_enc, pos_spiketrain = get_neuron_response_to_current(time, dt, current, 1)
    voltages_neg_enc, neg_spiketrain = get_neuron_response_to_current(time, dt, current, -1)


    x = current
    X = components
    X = np.roll(X, int(len(X)/2)) # This is necessary so that the frequencies of F line up with the frequencies of ts
    frequencies = np.roll(frequencies, int(len(frequencies)/2)) # Just useful for comparing the frequencies of X_w to the frequencies expected by ts

    spikes = np.array([
        pos_spiketrain, 
        -1 * np.array(neg_spiketrain)
    ])


    ts, fs, R, H, h, XHAT, xhat, XP, WXP = compute_optimal_filter(x, X, spikes, dt=1e-3)
    
    plt.scatter(ts[B_plt_lim:-B_plt_lim], h[B_plt_lim:-B_plt_lim])
    plt.title("4F) Filter h in time domain, random signal bw_limit = " + str(limit_hz_) + " Hz")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Strength")
    plt.grid()
    plt.show()

# G

discussion_G = """
    As the bandwidth of the random signal increases, more high-frequency components are allowed into
    the signal. To accomodate the increase in rapid spikes, the filter itself must have more
    high-frequency components, hence the increase in spikes in the time representation of the 
    filter.
"""

print(discussion_G)






