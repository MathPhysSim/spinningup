import logging.config
import math
import pickle
import random
import random as rd
from math import pi, exp

import gym
import numpy as np
# 3rd party modules
from gym import spaces
from scipy.integrate import quad

import Environments.twissElement as twiss


class transportENV(gym.Env):
    """
    Define a simple transport environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self,naf_flag=False):
        self.total_counter = 0
        self.__version__ = "0.0.5"
        logging.info("TransportEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_TIME = 100
        self.curr_step = -1
        self.is_finalized = False
        self.mssb = twiss.twissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.mbb = twiss.twissElement(8.88303879, -0.374085208, 0.057623602, 1.912555325)
        self.bpm1 = twiss.twissElement(339.174497, -6.521184683, 2.078511443, 2.081365696)
        self.bpm2 = twiss.twissElement(30.82651983, 1.876067844, 0.929904474, 2.163823492)
        self.target = twiss.twissElement(7.976311944, -0.411639485, 0.30867161, 2.398031982)

        self.x0 = 0.
        self.mssb_angle_0 = 0.001
        self.mbb_angle_0 = -0.001
        self.mssb_angle = self.mssb_angle_0  # radian
        self.mbb_angle = self.mbb_angle_0

        # Define what the agent can do
        self.MAX_POS = 1
        low = np.array([-self.MAX_POS, -self.MAX_POS])
        high = np.array([self.MAX_POS, self.MAX_POS])
        self.action_space = spaces.Box(low=low, high=high)

        # Observation is the position
        self.MAX_POS = 0.5
        low = np.array([-self.MAX_POS, -self.MAX_POS])
        high = np.array([self.MAX_POS, self.MAX_POS])
        self.observation_space = spaces.Box(low=low, high=high)

        self.counter = 0
        self.seed()
        self.viewer = None
        self.state = None

        self.curr_episode = -1

        self.beam_pos = 0.0
        self.intensity_on_target = [0.0, 0.0]

        self.rewards = []
        self.states_1 = []
        self.states_2 = []
        self.actions = []
        self.states = []

        self.naf_flag = naf_flag
        self.test_flag = False

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        observable, reward, episode_over, info : tuple
            observable (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # if self.is_finalized:
        #    raise RuntimeError("Episode is done")
        self.curr_step += 1
        state, reward = self._take_action(action)

        return np.array(state), reward, self.is_finalized, {}

    def _take_action(self, action):
        mssb_delta, mbb_delta = action * 1e-3
        self.mssb_angle += mssb_delta
        self.mbb_angle += mbb_delta

        self.counter += 1
        remaining_steps = self.MAX_TIME - self.counter
        episode_is_over = (remaining_steps <= 0)

        # print("just before check ",self.intensity_on_target[0])
        state, reward = self._get_state_and_reward()
        if self.naf_flag:
            if (self.intensity_on_target[0] > 0.8):
                episode_is_over = True
            else:
                reward = (reward - 0.8)/100
        else:
            if (self.intensity_on_target[0] > 0.8):
                episode_is_over = True
            reward = (reward - 0.8)

        if episode_is_over:
            self.is_finalized = True  # abuse this a bit

        if not self.test_flag:
            self.total_counter += 1
            self.rewards[self.curr_episode].append(reward)
            self.states_1[self.curr_episode].append(state[0])
            self.states_2[self.curr_episode].append(state[1])
            self.actions[self.curr_episode].append(action)
            self.states[self.curr_episode].append(state)
        return state, reward

    def _get_reward(self, beam_pos):
        self.beam_pos = beam_pos
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)
        self.intensity_on_target = quad(
            lambda x: 1 / (sigma * (2 * pi) ** 0.5) *
                      exp((x - beam_pos) ** 2 / (-2 * sigma ** 2)), -3 * sigma, 3 * sigma)
        reward = self.intensity_on_target[0]
        return reward

    def reset(self, init_state=[]):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        if not (self.test_flag):
            self.curr_episode += 1
            self.rewards.append([])
            self.states_1.append([])
            self.states_2.append([])
            self.actions.append([])
            self.states.append([])
        self.is_finalized = False
        self.x0 = 0.

        if len(init_state) == 0:
            # [-0.000701,  0.001419]#
            init_state = np.array(np.random.uniform(-0.001, 0.001, 2))

        self.mssb_angle = init_state[0]  # np.random.uniform(-0.002, 0.002, 1)[0]
        self.mbb_angle = init_state[1]  # np.random.uniform(-0.002, 0.002, 1)[0]
        self.counter = 0
        self.state, reward = self._get_state_and_reward()
        self.init_state = self.state
        return np.array(self.state)

    def _get_state_and_reward(self):
        """Get the observation."""

        x1, px1 = twiss.transport(self.mssb, self.mbb, self.x0, self.mssb_angle)
        px1 += self.mbb_angle

        bpm1_x, bpm1_px = twiss.transport(self.mbb, self.bpm1, x1, px1)
        bpm2_x, bpm2_px = twiss.transport(self.bpm1, self.bpm2, bpm1_x, bpm1_px)
        target_x, target_px = twiss.transport(self.bpm2, self.target, bpm2_x, bpm2_px)

        reward = self._get_reward(target_x)
        self.reward = reward

        delta1 = rd.uniform(-2.e-5, 2e-5)
        delta2 = rd.uniform(-2.e-5, 2e-5)

        delta1 = 0.0
        delta2 = 0.0
        bpm1_x += delta1
        bpm2_x += delta2

        state = [np.round(bpm1_x, 3, None), np.round(bpm2_x, 3, None)]
        return state, reward

    def render(self, mode='human'):
        carheight = 2
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-20, 20, 0, 2.2)

            xs = np.linspace(-self.MAX_POS, self.MAX_POS, 100)
            ys = self._get_intensity_from_state(xs)
            xys = list(zip((xs), ys))

            self.intensity = rendering.make_polyline(xys)
            self.intensity.set_linewidth(4)
            self.viewer.add_geom(self.intensity)

            self.magnetic_trans = rendering.Transform()
            beam_pos = rendering.make_circle(carheight / 2.5)
            beam_pos.add_attr(self.magnetic_trans)
            beam_pos.set_color(0, .9, .0)
            self.viewer.add_geom(beam_pos)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
        pos = self.state
        self.magnetic_trans.set_translation((pos), (0))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        random.seed(seed)

    def saveTrainingData(self):
        pickle_filename = "checkpoints/trainingData.pkl"
        output = open(pickle_filename, 'wb')
        pickle.dump(self.states, output, -1)
        pickle.dump(self.actions, output, -1)
        pickle.dump(self.rewards, output, -1)
        output.close()

    def readTrainingData(self):
        pickle_filename = "checkpoints/trainingData.pkl"
        pkl_file = open(pickle_filename, 'rb')
        states = pickle.load(pkl_file)
        actions = pickle.load(pkl_file)
        rewards = pickle.load(pkl_file)
        pkl_file.close()
        return states, actions, rewards


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nb_steps = 250
    init_angles = []
    init_pos = []
    init_rewards = []
    env = transportENV()
    for i in range(nb_steps):
        env.reset()
        init_pos.append(env.state)
        init_rewards.append(env.reward)
        init_angles.append([env.mssb_angle, env.mbb_angle])

    init_pos.append([0., min(np.array(init_pos)[:, 1])])
    init_pos = np.array(init_pos)

    plt.scatter(init_pos[:-1, 0], init_pos[:-1, 1], c=init_rewards, alpha=0.1)
    plt.show()
