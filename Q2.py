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
from Q1_1 import generate_signal



get_ipython().magic('clear')
get_ipython().magic('reset -f')


alpha = 3.35
J_bias = 1.46
Trc = 20/1000
Tref = 2/1000
Vth = 1

#2A)

dt = 1 / 1000
T = 1
N = int(T / dt)

x_linspace = np.linspace(0, T, N)

for x in [0, 1]:

    v = 0
    voltages = []
    i = 0
    
    while i < len(x_linspace):
        if v >= Vth:
            v = 0
            i += Tref * 1000 # Scaled to ms, since that is our step size here
            voltages.append(0)
            voltages.append(0)
        else:
            voltages.append(v)
            i += 1
            
        J = alpha * x + J_bias
        v += dt * (J - v) / Trc
            
    
    plt.grid()
    plt.plot(x_linspace[0:100], voltages[0:100])
    plt.ylabel("Voltage")
    plt.xlabel("Time")
    plt.title("2A) Time-Voltage graph of neuron with input of " + str(x))
    
    plt.show()



#2B)

B = """
Discussion:

"""

print(B)

#2C)

limit = 30
power_desired = 0.5
T = 1
dt = 0.001
seed = 18945

times, currents, _, _ = generate_signal(T, dt, power_desired, limit, seed)

voltages = []
v = 0
i = 0
while i < len(times):
    
    if v >= Vth:
        v = 0
        i += Tref * 1000 # Scaled to ms, since that is our step size here
        voltages.append(0)
        voltages.append(0)
    else:
        voltages.append(v)
        i += 1
        
    if i < len(times):
        J = alpha * currents[int(i)] + J_bias
        v += dt * (J - v) / Trc



plt.grid()
plt.plot(times[0:500], currents[0:500], label = "Currents")
plt.plot(times[0:500], voltages[0:500], label = "Voltages")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage")
plt.title("2C) Random signal limited at " + str(limit) + " Hz used as input to neuron")
plt.legend()
plt.show()

# 2D)

plt.grid()
plt.plot(times[0:200], currents[0:200], label = "Currents")
plt.plot(times[0:200], voltages[0:200], label = "Voltages")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage")
plt.title("2C) Random signal limited at " + str(limit) + " Hz used as input to neuron")
plt.legend()
plt.show()


#2E)

E = """
    How to improve the accuracy of the model matching the equation?
    
    This is a bit of a glib answer, but you could decrease the step size (yes, more computation),
    and use a more powerful computer, thus not increasing computation time.
    
    Alternatively, you could ditch the fixed step size of the LIF neuron, and
    make the step size a function of the last known slope, so when the curve is 
    steeper, use tighter steps, and when the curve is less steep, use larger steps.
    This would require extrapolation between discrete points of the input current
    for non-constant currents though, which would add some more computational overheard.
    The extrapolation computational requirements could be minimized by just using linear
    extrapolation, or just by generating a current that has a very small step, and using
    the nearest point in the current to whatever point is needed by the voltage equation.
    
    
    
    
"""

print(E)