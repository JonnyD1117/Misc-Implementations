from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import scipy.io
import numpy as np
from matplotlib import pyplot as plt




if __name__ == "__main__":

    mat = scipy.io.loadmat("I_FUDS.mat")
    mat2 = scipy.io.loadmat("pulse.mat")


    I_fuds = mat["I"][0][:]
    # I_fuds = I_fuds[0][:]

    time_fuds = mat['time'][0][:]

    # time_fuds = time_fuds[0][:]

    I_pulse = mat2["I"][0][:]
    # I_pulse = I_pulse[0][:]
    time_pulse = mat2['time'][0][:]
    # time_pulse = time_pulse[0][:]

    # plt.plot(I_fuds)
    # plt.show()


    SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=time_pulse[-1].item(), timestep=1)

    [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
    time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
    = SPMe.sim(CC=False, zero_init_I=False, I_input= I_pulse, init_SOC=.5, plot_results=True)





