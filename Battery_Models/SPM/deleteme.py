import numpy as np
from ScottBatteryParams import *



def compute_Stoich_coef(state_of_charge):
    """
    Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
    """
    alpha = state_of_charge

    stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
    stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant
    return [stoi_n, stoi_p]

def compute_SOC(theta_n, theta_p):
    """
    Computes the value of the SOC from either (N or P) electrode given the current
    Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
    """
    SOC_n = ((theta_n - stoi_n0)/(stoi_n100 - stoi_n0))
    SOC_p = ((theta_p - stoi_p0)/(stoi_p100 - stoi_p0))

    return [SOC_n, SOC_p]

for j in np.arange(0, 1, .1):

    value1, value2 = compute_Stoich_coef(j)
    print('#############----RATIO----####################')
    print(value1)
    print(value2)
    print("&&&&&&&&&----SOC------&&&&&&&&&&&&&&&&&&&&&&&&&&")

    soc1, soc2 = compute_SOC(value1, value2)
    print(soc1)
    print(soc2)
    print("@@@@@@@@@@@@@@@---- END ----@@@@@@@@@@@@@@@@@")