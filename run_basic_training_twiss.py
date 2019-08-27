import Environments.transport_enviroment as transport
import tensorflow as tf
from spinup.algos.ddpg.ddpg import ddpg
from spinup.algos.ppo.ppo import ppo
from spinup.algos.td3.td3 import td3
from spinup.utils.run_utils import ExperimentGrid
import pandas as pd
import matplotlib.pyplot as plt

from spinup.algos.naf.naf import naf

env = transport.transportENV()
# env_fn = lambda : gym.make('Pendulum-v0')

env_fn = lambda: env

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
output_dir = 'logging/test'
logger_kwargs = dict(output_dir=output_dir, exp_name='ppo twiss')
agent = ppo(env_fn=env_fn, epochs=100, steps_per_epoch=10000, logger_kwargs=logger_kwargs)


plot_name = 'Stats'
name = plot_name
data = pd.read_csv(output_dir+'/progress.txt', sep="\t")

data.index = data['TotalEnvInteracts']
data_plot= data[['EpLen', 'MinEpRet', 'AverageEpRet']]
data_plot.plot(secondary_y=['MinEpRet', 'AverageEpRet'])

plt.title(name)
# plt.savefig(name + '.pdf')
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
