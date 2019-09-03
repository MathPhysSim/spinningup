import gym

from spinup.algos.sac.sac import sac
import PyQt5
import Environments.transportEnvOld as transport
import tensorflow as tf
from spinup.algos.ddpg.ddpg import ddpg
from spinup.algos.ppo.ppo import ppo
from spinup.algos.td3.td3 import td3
from spinup.algos.trpo.trpo import trpo
from spinup.utils.run_utils import ExperimentGrid
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Qt5Agg")
import numpy as np
from spinup.algos.naf.naf import naf

env = transport.transportENV()
env_fn = lambda: env
# env_fn = lambda : gym.make('Pendulum-v0')


# if __name__ == '__main__':
#
#     eg = ExperimentGrid(name='twiss-bench')
#     eg.add('env_fn', env_fn, '', True)
#     eg.add('seed', [10 * i for i in range(2)])
#     eg.add('epochs', 25)
#     eg.add('steps_per_epoch', 1000)
#     eg.add('start_steps', 100)
#     eg.add('ac_kwargs:hidden_sizes', [(400, 300)], 'hid')
#     eg.add('ac_kwargs:activation', [tf.nn.relu], '')
#     eg.run(ddpg, num_cpu=4, data_dir='logging/DDPG')

ac_kwargs = dict(hidden_sizes=[100, 100])
act_noise = 1
output_dir = 'logging/new_environment/sac/test'
logger_kwargs = dict(output_dir=output_dir, exp_name='twiss')
agent = sac(env_fn=env_fn, epochs=100, steps_per_epoch=1000, logger_kwargs=logger_kwargs,
            gamma=0.999, seed=456)

plot_name = 'Stats'
name = plot_name
data = pd.read_csv(output_dir+'/progress.txt', sep="\t")

data.index = data['TotalEnvInteracts']
data_plot= data[['EpLen', 'MinEpRet', 'AverageEpRet']]
data_plot.plot(secondary_y=['MinEpRet', 'AverageEpRet'])

plt.title(name)
# plt.savefig(name + '.pdf')
plt.show()

# plotting
print('now plotting')
rewards = env.rewards
states_1 = env.states_1
states_2 = env.states_2

iterations = []
finals = []
max_1 = []
max_2 = []
min_1 = []
min_2 = []
init_state_1 = []
init_state_2 = []

init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

for i in range(len(rewards)):
    iterations.append(len(rewards[i]))
    if (len(states_1[i]) > 0):
        max_1.append(np.amax(states_1[i]))
        min_1.append(np.amin(states_1[i]))
    else:
        max_1.append(0.0)
        min_1.append(0.0)
    if (len(states_2[i]) > 0):
        max_2.append(np.amax(states_2[i]))
        min_2.append(np.amin(states_2[i]))
    else:
        max_2.append(0.0)
        min_2.append(0.0)

    if (len(rewards[i]) > 0):
        finals.append(rewards[i][len(rewards[i]) - 1])
        # init_state_1.append(init_states.iloc[i, 0])
        # init_state_2.append(init_states.iloc[i, 1])
        init_state_1.append(states_1[i][0])
        init_state_2.append(states_2[i][0])
    else:
        finals.append(0.0)
        init_state_1.append(0.0)
        init_state_2.append(0.0)

plot_suffix = ', number of iterations: ' + str(env.total_counter)

plt.figure(1)
plt.subplot(311)
plt.ylim()
plt.plot(iterations)
plt.title('Iterations' + plot_suffix)

plt.subplot(312)
plt.plot(finals, 'r--')
plt.ylim(0, 1)
plt.title('Reward' + plot_suffix)

plt.subplot(313)
plt.plot(max_1, 'g--')
plt.plot(min_1, 'r--')
plt.plot(max_2, 'g-')
plt.plot(min_2, 'r-')
plt.ylim(-.1, .1)
plt.title("positions" + plot_suffix)

plt.figure(2)
plt.title("Coverage" + plot_suffix)
# plt.subplot(414)
plt.scatter(init_state_1, init_state_2, s=80, c=finals, marker='o')
plt.tight_layout()
plt.show()

# env, get_action = load_policy('path/logging1')

# run_policy(env, get_action, render=False, max_ep_len=200, num_episodes=100)

# nb_steps = 250
# init_angles = []
# init_pos = []
# init_rewards = []
#
# for i in range(nb_steps):
#     env.reset()
#     init_pos.append(env.state)
#     init_rewards.append(env.reward)
#     init_angles.append([env.mssb_angle, env.mbb_angle])
#
# init_pos.append([0., min(np.array(init_pos)[:, 1])])
# init_pos = np.array(init_pos)

# save_folder = 'Graphics/'
# prefix_name = save_folder + algorithm + ', nr training: ' + str(epochs * steps_per_epoch) + ', '
#
# plot_name = 'Stats'
# name = prefix_name + plot_name
# data = pd.read_csv('path/logging1/progress.txt', sep="\t")
# data.index = data['TotalEnvInteracts']
# data[['AverageEpRet', 'EpLen']].plot()
# plt.title(name)
# plt.savefig(name + '.pdf')
# plt.show()
#
# plot_name = 'Old'
# name = prefix_name + plot_name
# plt.scatter(init_pos[:-1, 0], init_pos[:-1, 1], c=init_rewards, alpha=0.1)
# plt.title(name)
# plt.savefig(name + '.pdf')
# plt.show()
#
#
# def get_reward_from_init_angle_of_trained_agent(init_angle):
#     state, reward, finished = env.reset(init_angle), 0, False
#     proposed_action = get_action(state)
#     n = 1
#     while not (finished):
#         proposed_action = get_action(state)
#         state, reward, finished, _ = env.step(proposed_action)
#         n += 1
#     return env.reward, n
#
#
# new_rewards = []
# time = []
# for i in range(nb_steps):
#     reward, n = get_reward_from_init_angle_of_trained_agent(init_angles[i])
#     new_rewards.append(reward)
#     time.append(n)
#
# new_rewards.append(0.)
#
# plt.scatter(init_pos[:, 0], init_pos[:, 1], c=new_rewards, alpha=0.1)
# plot_name = 'New'
# name = prefix_name + plot_name
# plt.title(name)
# plt.savefig(name + '.pdf')
# plt.show()
#
# plot_name = 'Test reward'
# name = prefix_name + plot_name
# plt.plot(new_rewards)
# plt.title(name)
# plt.savefig(name + '.pdf')
# plt.show()
#
# plot_name = 'Time'
# name = prefix_name + plot_name
# plt.plot(time)
# plt.title(name)
# plt.savefig(name + '.pdf')
# plt.show()
