import numpy as np
from SPMeBatteryParams import *

# from numpy import tanh, exp
from math import tanh, exp

v1 = np.zeros([1, 20])
v2 = np.zeros([1, 20])
v3 = np.zeros([1, 20])
v4 = np.zeros([1, 20])
v5 = np.zeros([1, 20])
v6 = np.zeros([1, 20])
v7 = np.zeros([1, 20])
v8 = np.zeros([1, 20])
v9 = np.zeros([1, 20])
v10 = np.zeros([1, 20])

print("indiv v", np.shape(v1[0, :]))
clip_val = 10

vect = np.array([v1[0, :], v2[0, :], v3[0, :], v4[0, :], v5[0, :], v6[0, :], v7[0, :], v8[0, :], v9[0, :], v10[0, :]])

print("Vect Shape", np.shape(vect))

# vect = np.zeros([10, 20])
print(vect)

thing = np.ones([10, 1])
print(thing)

vect[:, 0] = thing[:, 0]

print(vect)





print("#####################################")
print("#####################################")

cut_off = 10

[n, m] = np.shape(vect)
vect = vect[:, :10]

print(vect)