from SPMeBatteryParams import *
from math import asinh, tanh, cosh
import numpy as np


theta_p = .5
k_p = .5
cep = 1000
cs_max_p = 5.1219e+04
Jp = .5
j0_p = .5
out_Sepsi_p = .5
docvp_dCsep = .5



rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ** 2 + 1) ** 0.5)) * (1 + k_p / ((k_p ** 2 + 1) ** 0.5)) * (
                -3 * Jp / (2 * as_p ** 2 * j0_p * Rp))

rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (
                    cep * theta_p * (cs_max_p - theta_p)) * (1 + 1 / (k_p) ** 2) ** (-0.5)

sen_out_spsi_p = (rho1p + (rho2p + docvp_dCsep) * -out_Sepsi_p)

print(rho1p)
print(rho2p)
print(sen_out_spsi_p)