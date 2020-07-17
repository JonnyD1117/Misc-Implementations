import numpy as np
from SPMeBatteryParams import *

# from numpy import tanh, exp
from math import tanh, exp


def OCV_Anode(theta):
    # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
    # Uref1 = 0.194 + 1.5 * exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh((theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh((theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh((theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)
    Uref1 = 0.194 + 1.5 * np.exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh(
        (theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh(
        (theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh(
        (theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)
    Uref2 = 0.194 + 1.5 * exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh((theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh((theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh((theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)

    return [Uref1, Uref2]
def compute_Stoich_coef(state_of_charge):
    """
    Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
    """
    alpha = state_of_charge

    stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
    stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant
    return [stoi_n, stoi_p]

# print(compute_Stoich_coef(1))

x = np.arange(.442, .676, .001)

print(x[1])
print(len(x))
print("---------------------")



for i in range(0, len(x)):
    val = x[i]
    print(OCV_Anode(val))

