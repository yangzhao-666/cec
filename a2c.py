import gym
import numpy as np
from stable_baselines3 import A2C

import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
#import mujoco_maze

from wrapper import NoTimeStepNoVelocityMazeEnv
from wrapper import SparseMazeEnv

import os

def train(config, wandb_session):

    env = gym.make(config.env)
    if 'Maze' in config.env or 'Room' in config.env:
        env = SparseMazeEnv(env)
        env = NoTimeStepNoVelocityMazeEnv(env)
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{wandb_session.id}", n_steps=config.n_steps, gamma=config.gamma, learning_rate=config.lr, device='auto')
    model.learn(total_timesteps=config.num_steps, eval_freq=config.eval_freq, callback=WandbCallback(verbose=2), eval_env=env)
    model.save('./saved_model/a2c')

if __name__ == "__main__":
    description = 'A2C'
    parser = argparse.ArgumentParser(description=description)
    #environment settings
    parser.add_argument('--project', type=str, default='A2C')
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    #parser.add_argument('--env', type=str, default='Point4Rooms-v2')
    #parser.add_argument('--env', type=str, default='PointUMaze-v1')
    #parser.add_argument('--env', type=str, default='SwimmerUMaze-v1')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=16)
    parser.add_argument('--record', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')

    args = parser.parse_args()

    for run in range(args.runs):
        if args.wandb:
            #wandb_session = wandb.init(project=args.env, config=vars(args), name="run-%i"%(run), reinit=True, group=args.mode)
            wandb_session = wandb.init(project=args.env + '_A2C', config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False, sync_tensorboard=True)
        else:
            wandb_session = wandb.init(project=args.env + '_A2C', config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')


        config = wandb.config
        train(config, wandb_session)
