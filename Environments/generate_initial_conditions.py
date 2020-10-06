import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Environments.transportEnvOld import transportENV

np.random.seed(999)
dataInit = pd.DataFrame()
dataInit['states1'] = np.zeros(1000)
dataInit['states2'] = np.zeros(1000)
for i in range(1000):
    ran = min(float(0.002 *i), 0.002)
    print(ran)
    dataInit.iloc[i,:] = np.array(np.random.uniform(-ran, ran ,2))

dataInit.to_pickle('initData')

pd.read_pickle('initData').reset_index().plot.scatter(x='states1', y='states2', c='index')
plt.show()

nb_steps = 250
init_angles = []
init_pos = []
init_rewards = []
env = transportENV()
for i in range(nb_steps):
    env.reset(dataInit.iloc[i,:])
    init_pos.append(env.state)
    init_rewards.append(env.reward)
    init_angles.append([env.mssb_angle, env.mbb_angle])

init_pos.append([0., min(np.array(init_pos)[:, 1])])
init_pos = np.array(init_pos)

plt.scatter(init_pos[:-1, 0], init_pos[:-1, 1], c=init_rewards, alpha=0.1)
plt.show()