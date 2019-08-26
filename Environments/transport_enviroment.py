import logging.config
import math
import random
from math import pi, exp

import gym
import numpy as np
# 3rd party modules
from gym import spaces
from numpy import linalg as LA
from scipy.integrate import quad



class transportENV(gym.Env):
    """
    Define a simple Banana environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, scan_flag=False):
        self.scan_flag = scan_flag
        self.start_nr = 0
        self.criteria = False
        self.visualize = False
        # self.__version__ = "0.0.2"
        # logging.info("TransportEnv - Version {}".format(self.__version__))

        # General variables defining the environment

        self.MAX_TIME = 50
        self.curr_step = -1
        self.is_finalized = False
        self.mssb = twissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.mbb = twissElement(8.88303879, -0.374085208, 0.057623602, 1.912555325)
        self.bpm1 = twissElement(339.174497, -6.521184683, 2.078511443, 2.081365696)
        self.bpm2 = twissElement(30.82651983, 1.876067844, 0.929904474, 2.163823492)
        self.target = twissElement(7.976311944, -0.411639485, 0.30867161, 2.398031982)

        self.x0 = 0.
        self.mssb_angle_0 = 0.001
        self.mbb_angle_0 = -0.001
        self.mssb_angle = self.mssb_angle_0  # radian
        self.mbb_angle = self.mbb_angle_0

        # Observation is the position
        self.MAX_POS = 1e-1
        low = np.array([-self.MAX_POS, -self.MAX_POS])
        high = np.array([self.MAX_POS, self.MAX_POS])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.MAX_POS = 1
        low = np.array([-self.MAX_POS, -self.MAX_POS])
        high = np.array([self.MAX_POS, self.MAX_POS])
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        self.counter = 0
        self.seed()
        self.viewer = None
        # self.state = None

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        # self.OFFSET = 6

        self.beam_pos = 0.0
        self.intensity_on_target = [0.0, 0.0]

        self.seed(888)


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

        # print("reward, reward ",reward)

        # reward -=self.counter*0.000001
        return np.array(state), reward, self.is_finalized, {}

    def _take_action(self, action):
        action*=1e-4
        self.counter += 1
        remaining_steps = self.MAX_TIME - self.counter

        # self.action_episode_memory[self.curr_episode].append(action)
        mssb_delta, mbb_delta = action
        self.mssb_angle += mssb_delta
        self.mbb_angle += mbb_delta

        state, reward = self._get_state_and_reward()
        self.state = state
        # state_large = LA.norm(self.state) > 0.08
        # action_large = LA.norm(action) > 1e-3
        # if state_large or action_large:
        #     # reward = -.01
        #     # reward = -LA.norm(state) ** 2 * 1e2
        #     # print("State: ", self.state, action)
        #     self.is_finalized = True
        #     pass



        if (self.intensity_on_target[0] > 0.95):

            self.is_finalized = True
            if self.scan_flag:
                pass
                # print('pass')
            else:
                reward = 10
        else:
            if self.scan_flag:
                pass
                # print('pass')
            else:
                reward = -1e-2
            pass
        self.start_nr += 1

        if remaining_steps < 0:
            self.is_finalized = True

        return state, reward

    def _get_reward(self, beam_pos):
        # print("beam_pos ", beam_pos)

        self.beam_pos = beam_pos
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)

        self.intensity_on_target = (quad(lambda x: 1 / (sigma * (2 * pi) ** 0.5)
                                                   * exp((x - beam_pos) ** 2 / (-2 * sigma ** 2)), -3 * sigma,
                                         3 * sigma,
                                         ))

        reward = self.intensity_on_target[0]

        return reward

    def set_criteria(self, flag):
        self.criteria = flag

    def reset(self, initial_angles=False):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_finalized = False
        self.x0 = 0.
        self.counter = 0
        self.start_nr += 1

        initial_variation = 2e-3

        if not (initial_angles):
            while self.criteria:
                self._set_init(initial_variation)
                state_large = LA.norm(self.state) < 0.005
                if not (state_large):
                    break
            if not (self.criteria):
                self._set_init(initial_variation)
        else:
            self.mssb_angle, self.mbb_angle = initial_angles
            self.state, self.reward = self._get_state_and_reward()
        # print("state: ", self.state, reward)

        return np.array(self.state)

    def _set_init(self, initial_variation):
        self.mssb_angle = np.random.uniform(-initial_variation, initial_variation, 1)[0]
        self.mbb_angle = np.random.uniform(-initial_variation, initial_variation, 1)[0]
        state, reward = self._get_state_and_reward()
        self.state = state

    def _get_state_and_reward(self):
        """Get the observation."""

        x1, px1 = transport(self.mssb, self.mbb, self.x0, self.mssb_angle)
        px1 += self.mbb_angle

        bpm1_x, bpm1_px = transport(self.mbb, self.bpm1, x1, px1)
        bpm2_x, bpm2_px = transport(self.bpm1, self.bpm2, bpm1_x, bpm1_px)
        target_x, target_px = transport(self.bpm2, self.target, bpm2_x, bpm2_px)

        self.reward = self._get_reward(target_x)

        # if(np.abs(self.beam_pos)>0.05):
        #    reward = -10.
        noise = False
        if noise:
            delta1, delta2 = np.random.normal(0.0, 1e-4, 2)
        else:
            delta1, delta2 = 0, 0
        self.state = [bpm1_x + delta1, bpm2_x + delta2]

        return self.state, self.reward

    def render(self, mode='human'):
        clearance = 1
        carheight = carwidth = 2
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-20, 20, 0, 2.2)

        # self.pole_transform.set_rotation(self.state[0] + np.pi / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        np.random.seed(seed)

import math


class twissElement():

    def __init__(self, beta, alpha, d, mu):
        self.beta = beta
        self.alpha = alpha
        self.mu = mu


def transport(element1, element2, x, px):
    mu = element2.mu - element1.mu
    alpha1 = element1.alpha
    alpha2 = element2.alpha
    beta1 = element1.beta
    beta2 = element2.beta

    m11 = math.sqrt(beta2 / beta1) * (math.cos(mu) + alpha1 * math.sin(mu))
    m12 = math.sqrt(beta1 * beta2) * math.sin(mu)
    m21 = ((alpha1 - alpha2) * math.cos(mu) - (1 + alpha1 * alpha2) * math.sin(mu)) / math.sqrt(beta1 * beta2)
    m22 = math.sqrt(beta1 / beta2) * (math.cos(mu) - alpha2 * math.sin(mu))

    return m11 * x + m12 * px, m21 * x + m22 * px
