import gym
import tensorflow as tf

import spinup
from spinup.algos.naf.naf import naf

env_fn = lambda : gym.make('Pendulum-v0')

ac_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

steps_per_epoch = 100
epochs = 10
start_steps = 100
algorithm = 'naf - tests'


# agent = naf(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=100, logger_kwargs=logger_kwargs)

agent = spinup.ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=250, logger_kwargs=logger_kwargs,
                    start_steps=start_steps)