import gym
import tensorflow as tf
# from spinup.algos.ddpg.ddpg import ddpg

from spinup.utils.run_utils import ExperimentGrid

import spinup
from spinup.algos.naf.naf import naf

env_fn = lambda : gym.make('Pendulum-v0')

ac_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='logging/NAF', exp_name='naf - tests')

steps_per_epoch = 100
epochs = 10
start_steps = 100
algorithm = 'naf - tests'

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=4)
    args = parser.parse_args()

    eg = ExperimentGrid(name='naf-bench-long')
    eg.add('env_name', 'Pendulum-v0', '', True)
    eg.add('seed', [10 * i for i in range(args.num_runs)])
    eg.add('epochs', 50)
    eg.add('steps_per_epoch', 200)
    eg.add('ac_kwargs:hidden_sizes', [(100, 100)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(naf, num_cpu=args.cpu, data_dir='logging/NAF')

# agent = naf(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=100, epochs=25, logger_kwargs=logger_kwargs)

# agent = spinup.ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=250, logger_kwargs=logger_kwargs,
#                     start_steps=start_steps)