from ScottBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt

from math import asinh, tanh




class SingleParticleModel(np):
    def __init__(self, timestep=1, sim_duration=1300, default_input=True, initial_SOC=.5, C_Rate="1C" ):
        super(SingleParticleModel, self).__init__()

        self.dt = timestep
        self.duration = sim_duration
        self.time = np.arange(0, self.duration, self.dt)

        self.base_current = -25.67              # Base Current Draw
        self.SOC_0 = initial_SOC

        self.C_rate = C_Rate
        self.C_rate_list = {"1C": 3601, "2C": 1712, "3C": 1083, "Qingzhi_C": 1300}

        self.default_input = default_input
        self.CC_input_profile = self.base_current*np.ones(self.C_rate_list["Qingzhi"])

        # Model Parameters & Variables

        # Negative electrode three-state continuous time state space model for the particle
        self.An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (Ds_n ** 2) / Rn ** 4), - (189 * Ds_n / Rn ** 2)]])
        self.Bn = np.array([[0], [0], [-1]])
        self.Cn = rfa_n * np.array([[10395 * Ds_n ** 2, 1260 * Ds_n * Rn ** 2, 21 * Rn ** 4]])
        self.Dn = np.array([0])

        # Approximate Negative Electrode Discretization
        Ts = self.dt
        [n, m] = np.shape(self.An)
        self.A_dn = np.eye(n) + self.An * Ts
        self.B_dn = self.Bn * Ts
        self.C_dn = self.Cn
        self.D_dn = self.Dn

        # discharge
        # Populate State Variables with Initial Condition
        xn = np.zeros([3, Kup + 1])
        xp = np.zeros([3, Kup + 1])
        yn = np.zeros(Kup + 1)
        yp = np.zeros(Kup + 1)
        theta_n = np.zeros(Kup)
        theta_p = np.zeros(Kup)
        V_term = np.zeros(Kup)
        time = np.zeros(Kup)

        xn[:, [0]] = np.array([[stoi_x * cs_max_n / (rfa_n * 10395 * Ds_n ** 2)], [0],
                               [0]])  # stoi_x100 should be changed if the initial soc is not equal to 50 %
        xp[:, [0]] = np.array([[stoi_y * cs_max_p / (rfa_p * 10395 * Ds_p ** 2)], [0],
                               [0]])  # initial positive electrode ion concentration

        # Positive electrode three-state state space model for the particle
        Ap = 1 * np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (Ds_p ** 2) / Rp ** 4), - (189 * Ds_p / Rp ** 2)]])
        Bp = np.array([[0], [0], [1]])
        Cp = rfa_p * np.array([[10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4]])
        Dp = np.array([0])

        # Approximate Positive Electrode Discretization
        [n, m] = np.shape(Ap)
        A_dp = np.eye(n) + Ap * Ts
        B_dp = Bp * Ts
        C_dp = Cp
        D_dp = Dp

        # Model Initialization



    def OCV_Anode(self, theta):
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


    def OCV_Cathod(self, theta):
        Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta)
        + 2.1581 * tanh(52.294 - 50.294 * theta)
        - 0.14169 * tanh(11.0923 - 19.8543 * theta)
        + 0.2051 * tanh(1.4684 - 5.4888 * theta)
        + 0.2531 * tanh((-theta + 0.56478) / 0.1316)
        - 0.02167 * tanh((theta - 0.525) / 0.006)

        return Uref

    def sim(self, I_input, SOC_0):
        """
        sim function runs complete solution given a timeseries current profile
        :return: [Terminal Voltage (time series), SOC (time Series) Input Current Profile (time series) ]
        """

        if self.default_input:




        else:


        for k in range(0, self):




            [] = self.step()
            """time[k] = k

            # Molar Current Flux Density (Assumed UNIFORM for SPM)
            Jn = I[k] / Vn
            Jp = I[k] / Vp

            # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
            yn[k] = C_dn @ xn[:, k] + D_dn * 0
            yp[k] = C_dp @ xp[:, k] + D_dp * 0

            # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
            xn[:, [k + 1]] = A_dn @ xn[:, [k]] + B_dn * Jn
            xp[:, [k + 1]] = A_dp @ xp[:, [k]] + B_dp * Jp

            if k == 1:
                print(xp[:, [k + 1]])

            # Compute "Exchange Current Density" per Electrode (Pos & Neg)
            i_0n = kn * F * (cen ** .5) * ((yn[k]) ** .5) * ((cs_max_n - yn[k]) ** .5)
            i_0p = kp * F * (cep ** .5) * ((yp[k]) ** .5) * ((cs_max_p - yp[k]) ** .5)

            # Compute Electrode "Overpotentials"
            eta_n = ((2 * R * T) / F) * asinh((Jn * F) / (2 * i_0n))
            eta_p = ((2 * R * T) / F) * asinh((Jp * F) / (2 * i_0p))

            # Record SOC of Cell
            theta_n[k] = yn[k] / cs_max_n
            theta_p[k] = yp[k] / cs_max_p

            U_n = self.OCV_Anode(theta_n[k])
            U_p = self.OCV_Cathod(theta_p[k])

            V_term[k] = U_p - U_n + eta_p - eta_n"""
        return

    def step(self, I_input):
        """
        step function runs one iteration of the model given the input current and
        :return:
        """
        time[k] = k

        # Molar Current Flux Density (Assumed UNIFORM for SPM)
        Jn = I[k] / Vn
        Jp = I[k] / Vp

        # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
        yn[k] = C_dn @ xn[:, k] + D_dn * 0
        yp[k] = C_dp @ xp[:, k] + D_dp * 0

        # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
        xn[:, [k + 1]] = A_dn @ xn[:, [k]] + B_dn * Jn
        xp[:, [k + 1]] = A_dp @ xp[:, [k]] + B_dp * Jp

        if k == 1:
            print(xp[:, [k + 1]])

        # Compute "Exchange Current Density" per Electrode (Pos & Neg)
        i_0n = kn * F * (cen ** .5) * ((yn[k]) ** .5) * ((cs_max_n - yn[k]) ** .5)
        i_0p = kp * F * (cep ** .5) * ((yp[k]) ** .5) * ((cs_max_p - yp[k]) ** .5)

        # Compute Electrode "Overpotentials"
        eta_n = ((2 * R * T) / F) * asinh((Jn * F) / (2 * i_0n))
        eta_p = ((2 * R * T) / F) * asinh((Jp * F) / (2 * i_0p))

        # Record SOC of Cell
        theta_n[k] = yn[k] / cs_max_n
        theta_p[k] = yp[k] / cs_max_p

        U_n = self.OCV_Anode(theta_n[k])
        U_p = self.OCV_Cathod(theta_p[k])

        V_term[k] = U_p - U_n + eta_p - eta_n


        return

# Simulation Time
Kup = 1300
# C-Rates (battery Charge/Discharge Rate)
# 3601sec # 1C
# 1712sec # 2C
# 1083sec # 3C              ;

I = np.zeros(Kup)
#  Generate/load pulse profile
for k in range(0, Kup):
    if k == 0:
        I[k] = 0
    else:
        # I(k) = 1
        # I(k) = 25.5;
        I[k] = -25.67 * 3

Kup = len(I)












plt.figure(1)
plt.title("Terminal Voltage vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Volts")
plt.plot(time,V_term)

plt.figure(2)
plt.title("Input Current vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Current")
plt.plot(time,I)

plt.figure(3)
plt.title("SOC vs time")
plt.xlabel("Time [sec]")
plt.ylabel("State of Charg")
plt.plot(time,theta_n)
plt.show()