from ScottBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt
from math import asinh, tanh


class SingleParticleModel:
    def __init__(self, timestep=1, sim_duration=1300, initial_SOC=.5, C_Rate="1C"):
        # super(SingleParticleModel, self).__init__()

        # Simulation Settings
        self.dt = timestep
        self.duration = sim_duration
        self.time = np.arange(0, self.duration, self.dt)
        self.num_steps = len(self.time)
        Ts = self.dt

        # Default Input "Current" Settings
        self.default_current = -25.67*3              # Base Current Draw
        self.SOC_0 = initial_SOC

        self.C_rate = C_Rate
        self.C_rate_list = {"1C": 3601, "2C": 1712, "3C": 1083, "Qingzhi_C": 1300}

        self.CC_input_profile = self.default_current*np.ones(self.C_rate_list["Qingzhi_C"]+1)
        self.CC_input_profile[0] = 0

        # Model Parameters & Variables
        ###################################################################
        # Positive electrode three-state state space model for the particle
        self.Ap = 1 * np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (Ds_p ** 2) / Rp ** 4), - (189 * Ds_p / Rp ** 2)]])
        self.Bp = np.array([[0], [0], [1]])
        self.Cp = rfa_p * np.array([[10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4]])
        self.Dp = np.array([0])

        # Positive electrode SS Discretized
        [n_pos, m_pos] = np.shape(self.Ap)
        self.A_dp = np.eye(n_pos) + self.Ap * Ts
        self.B_dp = self.Bp * Ts
        self.C_dp = self.Cp
        self.D_dp = self.Dp

        # Negative electrode three-state state space model for the particle
        self.An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (Ds_n ** 2) / Rn ** 4), - (189 * Ds_n / Rn ** 2)]])
        self.Bn = np.array([[0], [0], [-1]])
        self.Cn = rfa_n * np.array([[10395 * Ds_n ** 2, 1260 * Ds_n * Rn ** 2, 21 * Rn ** 4]])
        self.Dn = np.array([0])

        # Negative electrode SS Discretized
        [n_neg, m_neg] = np.shape(self.An)
        self.A_dn = np.eye(n_neg) + self.An * Ts
        self.B_dn = self.Bn * Ts
        self.C_dn = self.Cn
        self.D_dn = self.Dn

        # Model Initialization
    @staticmethod
    def OCV_Anode(theta):
        # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
        Uref = 0.194 + 1.5 * np.exp(-120.0 * theta)
        + 0.0351 * tanh((theta - 0.286) / 0.083)
        - 0.0045 * tanh((theta - 0.849) / 0.119)
        - 0.035 * tanh((theta - 0.9233) / 0.05)
        - 0.0147 * tanh((theta - 0.5) / 0.034)
        - 0.102 * tanh((theta - 0.194) / 0.142)
        - 0.022 * tanh((theta - 0.9) / 0.0164)
        - 0.011 * tanh((theta - 0.124) / 0.0226)
        + 0.0155 * tanh((theta - 0.105) / 0.029)

        return Uref

    @staticmethod
    def OCV_Cathod(theta):
        Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta)
        + 2.1581 * tanh(52.294 - 50.294 * theta)
        - 0.14169 * tanh(11.0923 - 19.8543 * theta)
        + 0.2051 * tanh(1.4684 - 5.4888 * theta)
        + 0.2531 * tanh((-theta + 0.56478) / 0.1316)
        - 0.02167 * tanh((theta - 0.525) / 0.006)

        return Uref

    @staticmethod
    def compute_Stoich_coef(state_of_charge):
        """
        Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
        """
        alpha = state_of_charge

        stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
        stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant
        return [stoi_n, stoi_p]

    @staticmethod
    def compute_SOC(theta_n, theta_p):
        """
        Computes the value of the SOC from either (N or P) electrode given the current
        Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
        """
        SOC_n = ((theta_n - stoi_n0)/(stoi_n100 - stoi_n0))
        SOC_p = ((theta_p - stoi_p0)/(stoi_p100 - stoi_p0))

        return [SOC_n, SOC_p]

    @staticmethod
    def plot_results():
        """plt.figure(1)
        plt.title("Terminal Voltage vs time")
        plt.xlabel("Time [sec]")
        plt.ylabel("Volts")
        plt.plot(time, V_term)

        plt.figure(2)
        plt.title("Input Current vs time")
        plt.xlabel("Time [sec]")
        plt.ylabel("Current")
        plt.plot(time, I)

        plt.figure(3)
        plt.title("SOC vs time")
        plt.xlabel("Time [sec]")
        plt.ylabel("State of Charg")
        plt.plot(time, theta_n)
        plt.show()"""

    def sim(self, I_input=None, SOC_0=None):
        """
        sim function runs complete solution given a timeseries current profile
        :return: [Terminal Voltage (time series), SOC (time Series) Input Current Profile (time series) ]
        """
        Kup = self.num_steps

        # Populate State Variables with Initial Condition
        xn = np.zeros([3, Kup + 1])
        xp = np.zeros([3, Kup + 1])
        yn = np.zeros(Kup + 1)
        yp = np.zeros(Kup + 1)
        theta_n = np.zeros(Kup)
        theta_p = np.zeros(Kup)
        V_term = np.zeros(Kup)
        time = np.zeros(Kup)
        input_cur_prof = np.zeros(Kup)


        # Set Initial Simulation (Step0) Parameters/Inputs
        if SOC_0 is not None:
            input_soc = SOC_0
        # else:
        #     # theta_n, theta_p = self.compute_Stoich_coef()
        #     # input_soc, _ = self.compute_SOC(theta_n[0], theta_p[0])
        #     input_soc = .5
        input_state = None

        if I_input is None:
            input_current = self.CC_input_profile[0]
        else:
            input_current = I_input

        # Main Simulation Loop
        for k in range(0, Kup):
            # Perform one iteration of simulation using "step" method
            states, soc_new, V_out, theta = self.step(input_state, input_current, input_soc)

            # Record Desired values for post-simulation plotting/analysis
            xn[:, [k]], xp[:, [k]], theta_n[k], theta_p[k], V_term[k], time[k], input_cur_prof[k] = states["xn"], states["xp"], theta[0].item(), theta[1].item(), V_out.item(), self.dt * k, input_current
            # Update "step"s inputs to continue and update the simulation
            input_state, input_soc, input_current = states, soc_new, self.CC_input_profile[k+1]

        return [xn, xp, theta_n, theta_p, V_term, time, input_cur_prof]

    def step(self, states=None, I_input=None, state_of_charge=None):
        """
        step function runs one iteration of the model given the input current and returns output states and quantities
        States: dict(), I_input: scalar, state_of_charge: scalar
        """
        # Create Local Copy of Discrete SS Matrices for Ease of notation when writing Eqns.
        A_dp = self.A_dp
        B_dp = self.B_dp
        C_dp = self.C_dp
        D_dp = self.D_dp

        A_dn = self.A_dn
        B_dn = self.B_dn
        C_dn = self.C_dn
        D_dn = self.D_dn

        # Initialize Input Current
        if I_input is None:
            I = self.default_current     # If no input signal is provided use CC @ default input value
        else:
            I = I_input

        # Initialize SOC
        if state_of_charge is None:
            soc = .5                    # If no SOC is provided by user then defaults to SOC = .5
        else:
            soc = state_of_charge

        # Initialize "State" Vector
        if states is None:
            stoi_n, stoi_p = self.compute_Stoich_coef(soc)

            print(stoi_n)

            # IF not initial state is supplied to the "step" method, treat step as initial step
            xn_old = np.array([[stoi_n * cs_max_n / (rfa_n * 10395 * Ds_n ** 2)], [0], [0]])  # stoi_n100 should be changed if the initial soc is not equal to 50 %
            xp_old = np.array([[stoi_p * cs_max_p / (rfa_p * 10395 * Ds_p ** 2)], [0], [0]])  # initial positive electrode ion concentration

            states = {"xn": xn_old, "xp": xp_old}
            outputs = {"yn": None, "yp": None}

        else:
            # ELSE use given states information to propagate model forward in time
            xn_old, xp_old = states["xn"], states["xp"]
            outputs = {"yn": None, "yp": None}

        # Molar Current Flux Density (Assumed UNIFORM for SPM)
        Jn = I / Vn
        Jp = -I / Vp

        # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
        yn_new = C_dn @ xn_old + D_dn * 0
        yp_new = C_dp @ xp_old + D_dp * 0

        outputs["yn"], outputs["yp"] = yn_new, yp_new

        # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
        xn_new = A_dn @ xn_old + B_dn * Jn
        xp_new = A_dp @ xp_old + B_dp * Jp

        states["xn"], states["xp"] = xn_new, xp_new

        # Compute "Exchange Current Density" per Electrode (Pos & Neg)
        i_0n = kn * F * (cen ** .5) * (yn_new ** .5) * ((cs_max_n - yn_new) ** .5)
        i_0p = kp * F * (cep ** .5) * (yp_new ** .5) * ((cs_max_p - yp_new) ** .5)

        # Compute Electrode "Overpotentials"
        eta_n = ((2 * R * T) / F) * asinh((Jn * F) / (2 * i_0n))
        eta_p = ((2 * R * T) / F) * asinh((Jp * F) / (2 * i_0p))

        # Record Stoich Ratio (SOC can be computed from this)
        theta_n = yn_new / cs_max_n
        theta_p = yp_new / cs_max_p

        theta = [theta_n, theta_p]   # Stoichiometry Ratio Coefficent

        soc_new = self.compute_SOC(theta_n, theta_p)

        U_n = self.OCV_Anode(theta_n)
        U_p = self.OCV_Cathod(theta_p)
        V_term = U_p - U_n + eta_p - eta_n

        return [states, soc_new, V_term, theta]


if __name__ == "__main__":

    SPM = SingleParticleModel()
    sim_out = SPM.step()
    print(sim_out)


    [step_states, step_soc_new, V_terminal, step_theta] = sim_out
    print(V_terminal)
    # [xn, xp, theta_n, theta_p, V_term, time, current] = SPM.sim()


    # plt.plot(time,V_term)
    # # plt.plot(time, current)
    # plt.show()
    #
    # print(V_term)
    # print(time)
