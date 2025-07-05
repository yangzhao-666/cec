import gym
from typing import Union
from gym import spaces

import numpy as np

#from SpaceX.constants import *

class NoTimeStepNoVelocityMazeEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(high=env.observation_space.high[:-4], low=env.observation_space.low[:-4])

    def reset(self):
        obs = self.env.reset()
        return obs[:-4]

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs[:-4], reward, done, info

class NoVelocityEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(high=env.observation_space.high[:-2], low=env.observation_space.low[:-2], shape=(env.observation_space.shape[0]-2, ))

    def reset(self):
        obs = self.env.reset()
        return obs[:-2]

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs[:-2], reward, done, info

# Directly add a random projection function on the top of envs. Currently, it is only for robotics envs.
class RandomProjectionRoboticsEnv(gym.core.Wrapper):
    def __init__(self, env, in_dim, out_dim=4):
        super().__init__(env)
        self.random_projection_matrix = np.random.normal(size=(in_dim, out_dim))

    def reset(self):
        obs = self.env.reset()
        s_g = np.concatenate((obs['observation'], obs['desired_goal']))
        projected_obs = np.matmul(s_g, self.random_projection_matrix)
        return projected_obs

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        s_g = np.concatenate((next_obs['observation'], next_obs['desired_goal']))
        projected_obs = np.matmul(s_g, self.random_projection_matrix)
        return projected_obs, reward, done, info

# Change step penalty to 0 and success reward to 1 for robotics envs. Also will terminate if the goal is reached. Originally, the agent will continually act until the maximum steps are reached.
class SparseRoboticsEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if reward == -1:
            reward = 0
        elif reward == 0:
            reward = 1
            done = True
        return next_obs, reward, done, info

# remove the step penalty.
class SparseMazeEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if reward == -0.0001:
            reward = 0
        return next_obs, reward, done, info

class RLRoboticsEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_space = gym.spaces.Box(high=env.observation_space.high, low=env.observation_space.low, shape=(env.observation_space.shape[0]-4, ))

    def reset(self):
        obs = self.env.reset()
        s_g = np.concatenate((obs['observation'], obs['desired_goal']))
        return s_g

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        s_g = np.concatenate((next_obs['observation'], next_obs['desired_goal']))
        return s_g, reward, done, info

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action, evaluate=False):
        """Steps through the environment and normalizes the observation."""
        obs, rews, dones, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs, evaluate)
        else:
            obs = self.normalize(np.array([obs]), evaluate)[0]
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        if not return_info:
            return obs
        else:
            return obs, info

    def normalize(self, obs, evaluate=False):
        """Normalises the observation using the running mean and variance of the observations."""
        if not evaluate:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

class RescaleAction(gym.ActionWrapper):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 1.0, 0.75])
        >>> env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> env.action_space
        Box(-0.5, [0.   0.5  1.   0.75], (4,), float32)
        >>> RescaleAction(env, min_action, max_action).action_space == gym.spaces.Box(min_action, max_action)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray],
        max_action: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action

# remove the step penalty.
class SparseMonCarEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = 0
        else:
            reward = 100

        return next_obs, reward, done, info

# remove the step penalty.
class SparseSpaceXEnv(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if self.env.landed_ticks == 60:
            reward = 1
        else:
            reward = 0

        return next_obs, reward, done, info

class RandomProjectionSafetyEnv(gym.core.Wrapper):
    def __init__(self, env, in_dim, out_dim=4):
        super().__init__(env)
        self.random_projection_matrix = np.random.normal(size=(in_dim, out_dim))

    def reset(self):
        obs = self.env.reset()
        projected_obs = np.matmul(obs, self.random_projection_matrix)
        return projected_obs

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        #if reward != 1:
        #    reward = 0
        projected_obs = np.matmul(next_obs, self.random_projection_matrix)
        return projected_obs, reward, done, info
