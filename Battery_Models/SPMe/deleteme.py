from SPMeBatteryParams import *


theta_p = .5
k_p = .5
cep = 1000
cs_max_p = 5.1219e+04



rho2p = (R * T) / (2 * 0.5 * F) * ((cep * cs_max_p - (2 * cep * theta_p)) / (cep * theta_p * (cs_max_p - theta_p))) * (1 + 1 / (k_p) ** 2) ** (-0.5)


print(rho2p)