from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity


if __name__ == "__main__":

    SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=1300)

    [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
     time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
        = SPMe.sim(CC=True, zero_init_I=True, I_input=[-25.67*3], init_SOC=0, plot_results=True)





