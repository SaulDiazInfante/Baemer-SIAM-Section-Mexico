import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6.180)


def rhs(y, t_zero):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    s_p_prime = beta * (l_p + i_p) - a * s_p * i_v
    l_p_prime = a * s_p * i_v - b * l_p - beta * l_p
    i_p_prime = b * l_p - beta * i_p
    s_v_prime = - Lambda * s_v * i_p - g * s_v + (1 - theta) * mu
    i_v_prime = Lambda * s_v * i_p - g * i_v + theta * mu
    rhs_np_array = np.array([s_p_prime, l_p_prime, i_p_prime, s_v_prime, i_v_prime])
    return (rhs_np_array)


beta = 0.01
a = 0.1
g = 0.02
mu = 0.3
theta = 0.2
b = 0.075
Lambda = 0.06

y_zero = np.array([0.9999, 0.00, 0.0001, 0.99, 0.01])
t = np.linspace(0, 70, 1000)
sol = odeint(rhs, y_zero, t)
df_01 = pd.read_csv("data_jegger_set01.csv")
df_02 = pd.read_csv("data_jegger_set02.csv")
#
plt.style.use('ggplot')
plt.scatter(df_01.time, df_01.proportion,
            s=80,
            marker='o',
            alpha=0.7,
            label='PSCL-4')
plt.scatter(df_02.time, df_02.proportion,
            s=80,
            marker='^',
            alpha=0.7,
            label='Rashmi')
plt.plot(t, sol[:, 2], 'k--', label='$I_p$')
plt.xlabel('$t$ (days)')
plt.ylabel('Infected plant proportion $I_p$')
plt.xlim(0, 70)
plt.legend(loc=0)
plt.show()
