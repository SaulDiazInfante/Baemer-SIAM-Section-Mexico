import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10, 6.180)


t = np.load('time.npy')
Cost_1 = np.load('one_control_cost.npy')
Cost_2 = np.load('two_control_cost.npy')
Cost_3 = np.load('three_control_cost.npy')
Cost_4 = np.load('only_infected_cost.npy')
plt.style.use('ggplot')
plt.plot(t,Cost_1, label= 'Fumigation')
plt.plot(t,Cost_2, label= 'Replanting plants')
plt.plot(t,Cost_3, label= 'Replanting and fumigation')
plt.plot(t,Cost_4, label= 'Replanting infected plant')
plt.xlabel('t(days)')
plt.ylabel('Cost')
plt.legend(loc=0)
plt.savefig('Cost_Comparation_version_2.pdf')
plt.show()