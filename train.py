import gym
import os
import argparse
import wandb
import numpy as np
import mujoco_maze
import matplotlib.pyplot as plt
import copy

from CEC import CEC
from wrapper import NoTimeStepNoVelocityMazeEnv
from wrapper import NoVelocityEnv
from wrapper import RandomProjectionRoboticsEnv
from wrapper import SparseRoboticsEnv
from wrapper import SparseMazeEnv
from wrapper import NormalizeObservation
from wrapper import SparseMonCarEnv
from wrapper import SparseSpaceXEnv
from wrapper import RandomProjectionSafetyEnv
#from SpaceX.SpaceX import RocketLander
#from SpaceX import FalconLander
import safety_gym
from safety_gym.envs.engine import Engine

from gym.wrappers import TimeLimit

from utils import plot_actions
from utils import plot_trace_actions
from utils import plot_retuns

from stable_baselines3 import SAC

def evaluate(CECagent, env, n_games=5, gt_model=None):
    rewards = []
    trajectory = []
    taken_steps = 0
    action_loss = 0
    state_distance = []
    total_steps = 0
    for n in range(n_games):
        s = env.reset()
        done = False
        reward = 0
        while not done:
            action, s_distance = CECagent.select_action(s, evaluate=True)
            total_steps += 1
            state_distance.append(s_distance)
            if gt_model is not None:
                gt_action, _ = gt_model.predict(s, deterministic=True)
                action_loss_step = (np.square(gt_action - action)).mean()
                action_loss += action_loss_step
            #n_s, r, done, _ = env.step(action, evaluate=True)
            n_s, r, done, _ = env.step(action)
            reward += r
            s = n_s
            trajectory.append(n_s[:2])
            taken_steps += 1
        rewards.append(reward)
    avg_steps = taken_steps / n_games
    print('CEC size: {} | avg eval steps taken: {} | action mse loss: {} | state distance: {}'.format(len(CECagent.returns), avg_steps, action_loss/total_steps, np.mean(state_distance)))
    return np.mean(rewards), trajectory, avg_steps, len(CECagent.returns), action_loss/total_steps, state_distance

def train(config, wandb_session):

    state_coverage = []
    max_return = 0
    if 'SpaceX' not in config.env and 'Safe' not in config.env:
        env = gym.make(config.env)
    if 'Maze' in config.env or 'Room' in config.env:
        env = SparseMazeEnv(env)
        env = NoTimeStepNoVelocityMazeEnv(env)
        env = NormalizeObservation(env)
    elif 'Fetch' in config.env or 'Hand' in config.env:
        obs = env.reset()
        indim = len(obs['observation']) + len(obs['desired_goal'])
        env = RandomProjectionRoboticsEnv(env, in_dim=indim, out_dim=config.random_projection_dim)
        env = SparseRoboticsEnv(env)
        env = NormalizeObservation(env)
        #if 'Hand' in config.env:
        #    raise ValueError('You are trying to use Hand env, but you didnt change the sparse reward setting.')
    elif 'Safe' in config.env:
        env_config = {
            'robot_base': 'xmls/point.xml',
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
        #env = NormalizeObservation(env)
        obs = env.reset()
        indim = len(obs)
        import ipdb;ipdb.set_trace()
        env = RandomProjectionSafetyEnv(env, in_dim=indim, out_dim=config.random_projection_dim)
    elif 'Car' in config.env:
        env = NormalizeObservation(env)
        env = SparseMonCarEnv(env)
    elif 'SpaceX' in config.env:
        """
        simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 500}
        env = RocketLander(simulation_settings)
        env = FalconLander()
        """
        env = gym.make("gym_rocketlander:rocketlander-v0")
        env = SparseSpaceXEnv(env)
    elif 'Swimmer-v2' in config.env:
        env = TimeLimit(env, max_episode_steps=config.episode_len)
    if config.record:
        record_env = gym.wrappers.Monitor(env, "./video", force=True, video_callable=lambda episode_id: True)
    CECagent = CEC(gamma=config.gamma, capacity=config.capacity, distance_threshold=config.distance_threshold, action_space=env.action_space, eps_start=config.eps_start, eps_end=config.eps_end, eps_decay_steps=config.eps_decay_steps, exploration=config.exploration, act_noise=config.act_noise, k=config.k, softmax_tau=config.tau, T=config.T)
    # load ground truth model if required
    if config.gt:
        if not os.path.exists(config.gt_file):
            raise ValueError('Ground truth file does not exist, plz check it.')
        gt_model = SAC('MlpPolicy', env)
        gt_model.load(config.gt_file)
    else:
        gt_model = None

    total_t = 0
    while total_t < config.num_steps:
        eps_states = []
        eps_actions = []
        eps_rewards = []
        s = env.reset()
        done = False
        while not done:
            action = CECagent.select_action(s)
            n_s, r, done, _ = env.step(action)
            total_t += 1
            eps_states.append(s)
            eps_actions.append(action)
            eps_rewards.append(r)
            state_coverage.append(s[:2])
            s = n_s
            if total_t % config.eval_freq == 0:
                env_copy = copy.deepcopy(env)
                if config.record:
                    eval_reward, trajectory, avg_steps, cec_len, action_loss, state_distance = evaluate(CECagent, record_env, gt_model=gt_model)
                else:
                    eval_reward, trajectory, avg_steps, cec_len, action_loss, state_distance = evaluate(CECagent, env_copy, gt_model=gt_model)
                del env_copy
                print('Maximum encountered return is: {}'.format(max_return))
                print('total steps: {} | eval rewrads: {} | eval avg steps taken: {} '.format(total_t, eval_reward, avg_steps))
                # CECagent.save('./')
                wandb_session.log({'eval rewards': eval_reward, 'training steps': total_t, 'eval steps taken': avg_steps, 'cec length': cec_len, 'avg state distance': np.mean(state_distance), 'eps': CECagent.epsilon})
                n_state_distance_larger = np.sum(np.array(state_distance) > config.distance_threshold)
                wandb_session.log({'number of larger state distance': n_state_distance_larger, 'training steps': total_t})
                if config.gt:
                    wandb_session.log({'action loss': action_loss, 'training steps': total_t})
                '''
                if total_t % (10 * config.eval_freq) == 0:
                    if 'Maze' in config.env or 'Room' in config.env:
                        x = [p[0] for p in trajectory]
                        y = [p[1] for p in trajectory]
                        fig = plt.scatter(x, y)
                        wandb_session.log({'eval trajectory': fig})

                        x = [p[0] for p in state_coverage]
                        y = [p[1] for p in state_coverage]
                        fig = plt.scatter(x, y)
                        wandb_session.log({'state coverage': fig})
                '''
                #if total_t % (1 * config.eval_freq) == 0 and 'Maze' in config.env:
                #fig = plot_actions(CECagent, env)
                #env_copy = copy.copy(env)
                #fig = plot_trace_actions(CECagent, env_copy)
                #fig = plot_retuns(CECagent, env_copy)
                #del env_copy
                #img = wandb.Image(fig)
                #wandb_session.log({'state coverage': img})

        G = CECagent.update(eps_states, eps_actions, eps_rewards)
        if G > max_return:
            max_return = G

if __name__ == "__main__":
    description = 'CEC'
    parser = argparse.ArgumentParser(description=description)
    #environment settings
    parser.add_argument('--project', type=str, default='SafetyGymCarRP')
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    #parser.add_argument('--env', type=str, default='Point4Rooms-v2')
    #parser.add_argument('--env', type=str, default='PointUMaze-v1')
    #parser.add_argument('--env', type=str, default='SwimmerUMaze-v1')
    #parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay_steps', type=int, default=100000)
    parser.add_argument('--distance_threshold', type=float, default=0.1)
    parser.add_argument('--capacity', type=int, default=1000000)
    parser.add_argument('--record', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--random_projection_dim', type=int, default=8)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--exploration', type=str, default='random')
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=1)

    parser.add_argument('--bs', default=False, action='store_true') # bootstrapping for the non-done terminal state, not implemented yet.

    # ground truth configs
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument('--gt_file', type=str, default='./saved_model/sac.zip')
    parser.add_argument('--episode_len', type=int, default=100)

    args = parser.parse_args()

    for run in range(args.runs):
        if args.wandb:
            #wandb_session = wandb.init(project=args.env, config=vars(args), name="run-%i"%(run), reinit=True, group=args.mode)
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')


        config = wandb.config
        train(config, wandb_session)
