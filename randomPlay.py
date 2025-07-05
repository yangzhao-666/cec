import gym
import numpy as np
import mujoco_maze
import safety_gym
from wrapper import RandomProjectionSafetyEnv
from safety_gym.envs.engine import Engine
import matplotlib.pyplot as plt

config = {
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'randomize_layout': False,
    'continue_goal': False,
    'goal_locations': [(2, 2)],
    'robot_locations': [(1, 1)],
    'reward_distance': 0.0,
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    '_seed': 1
}
#env = gym.make('InvertedPendulum-v2')
#env = gym.make('SwimmerUMaze-v1')
#env = gym.make('PointUMaze-v1')
#env = gym.make('FetchReach-v1')
#env = gym.make('Safexp-PointGoal0-v0')
#s = env.reset()
#env = RandomProjectionSafetyEnv(env, in_dim=len(s))
env = Engine(config)
import ipdb; ipdb.set_trace()

rewards = []
for i in range(100):
    done = False
    eps_reward = 0
    env.reset()
    while not done:
        action = env.action_space.sample()
        n_s, reward, done, _ = env.step(action)
        print('state: {}'.format(n_s))
        eps_reward += reward
        print('current step reward is {}'.format(reward))
    rewards.append(eps_reward)
    print('{} th run, episode reward is {}'.format(i, eps_reward))
print('eval reward: {}'.format(np.mean(rewards)))
