import gym
import gym.spaces
import sparse_gym_mujoco
import numpy as np
import torch
from stable_baselines3 import SAC

import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
#import mujoco_maze

from wrapper import NoTimeStepNoVelocityMazeEnv
from wrapper import SparseMazeEnv
from wrapper import SparseRoboticsEnv
from wrapper import RLRoboticsEnv
from wrapper import SparseMonCarEnv

from gym.wrappers import TimeLimit

import mujoco_maze
#import safety_gym
#from safety_gym.envs.engine import Engine

import os

def train(config, wandb_session):

    if 'Maze' in config.env or 'Room' in config.env:
        env = gym.make(config.env)
        env = SparseMazeEnv(env)
        env = NoTimeStepNoVelocityMazeEnv(env)
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    elif 'Fetch' in config.env:
        #env = RLRoboticsEnv(env)
        env = gym.make(config.env)
        env = SparseRoboticsEnv(env)
        model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    elif 'Car' in config.env:
        env = gym.make(config.env)
        env = SparseMonCarEnv(env)
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    #model = SAC('MlpPolicy', env, verbose=1, batch_size=1, buffer_size=1, tensorboard_log=f"runs/{wandb_session.id}")
    elif 'Safe' in config.env:
        env_config = {
            'robot_base': 'xmls/car.xml',
            'task': 'goal',
            'randomize_layout': False,
            'continue_goal': False,
            'goal_locations': [(2.4, 1.5)],
            'robot_locations': [(1, 1)],
            'reward_distance': 0.0,
            'goal_size': 0.2,
            'goal_keepout': 0.305,
            'hazards_size': 0.2,
            'hazards_keepout': 0.18,
            'walls_num': 44,
            'walls_locations': [(1.6, 1.5), (1.8, 1.5), (1.9, 1.5), (0.5, 0.5), (0.5, 0.7), (0.5, 0.9), (0.5, 1.1), (0.5, 1.3), (0.5, 1.5), (0.5, 1.7), (0.5, 1.9), (0.5, 2.1), (0.5, 2.2), (0.5, 2.4), (0.7, 2.4), (0.9, 2.4), (1.1, 2.4), (1.3, 2.4), (1.5, 2.4), (1.7, 2.4), (1.9, 2.4), (2.1, 2.4), (2.3, 2.4), (2.5, 2.4), (2.7, 2.4), (0.7, 0.5), (0.9, 0.5), (1.1, 0.5), (1.3, 0.5), (1.5, 0.5), (1.7, 0.5), (1.9, 0.5), (2.1, 0.5), (2.3, 0.5), (2.5, 0.5), (2.7, 0.5), (2.9, 0.5), (2.9, 0.7), (2.9, 0.9), (2.9, 1.1), (2.9, 1.3), (2.9, 1.5), (2.9, 1.7), (2.9, 1.9)],
            'walls_size': 0.2,
            '_seed': 1,
            # observation sensor
            'observe_walls': True,
            'observe_goal_comp': True
        }
        env = Engine(env_config)
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    elif 'Swimmer-v2' in config.env or 'Walker2d-v2' in config.env or 'HalfCheetah-v2' in config.env or 'Hopper-v2' in config.env:
        env = gym.make(config.env)
        env = TimeLimit(env, max_episode_steps=config.episode_len)
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    elif 'SparseHopper' in config.env or 'SparseHalfCheetah' in config.env or 'SparseWalker2d' in config.env or 'SparseSwimmer' in config.env or 'SparseAnt' in config.env:
        env = gym.make(config.env)
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", batch_size=config.batch_size, gamma=config.gamma, learning_rate=config.lr)
    torch.set_num_threads(5)
    model.learn(total_timesteps=config.num_steps, eval_freq=config.eval_freq, callback=WandbCallback(verbose=2), eval_env=env)
    #model.save('./saved_model/sac')

if __name__ == "__main__":
    description = 'SAC'
    parser = argparse.ArgumentParser(description=description)
    #environment settings
    parser.add_argument('--project', type=str, default='SAC_mujoco')
    #parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    #parser.add_argument('--env', type=str, default='Point4Rooms-v2')
    parser.add_argument('--env', type=str, default='PointUMaze-v1')
    #parser.add_argument('--env', type=str, default='SwimmerUMaze-v1')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--record', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')

    parser.add_argument('--episode_len', type=int, default=100)

    args = parser.parse_args()

    for run in range(args.runs):
        if args.wandb:
            #wandb_session = wandb.init(project=args.env, config=vars(args), name="run-%i"%(run), reinit=True, group=args.mode)
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False, sync_tensorboard=True)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')


        config = wandb.config
        train(config, wandb_session)
