import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Arrow

def softmax(x, tau=1):
    return np.exp(x/tau)/sum(np.exp(x/tau))

class ActionRescaler():
    def __init__(self, action_high, action_low):
        self.action_high = action_high
        self.action_low = action_low
        self.n = 1 / self.action_high

    def rescale_action(self, action):
    # rescale actions to range [-1, 1]
        scaled_action = action * self.n
        return scaled_action

    def scale_back_action(self, scaled_action):
        action = scaled_action / self.n
        return action

def plot_actions(cec, env):

    fig, ax = plt.subplots()
    width = (env.observation_space.low[0], env.observation_space.high[0])
    height = (env.observation_space.low[1], env.observation_space.high[1])
    ax.set_xlim(width)
    ax.set_ylim(height)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for x in np.arange(int(env.observation_space.low[0]), int(env.observation_space.high[0])):
        for y in np.arange(int(env.observation_space.low[1]), int(env.observation_space.high[1])):
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='white'))
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))
    ax.axvline(width[0],0,height[1],linewidth=5,c='k')
    ax.axvline(width[1],0,height[1],linewidth=5,c='k')
    ax.axhline(width[0],0,width[1],linewidth=5,c='k')
    ax.axhline(height[1],0,width[1],linewidth=5,c='k')
    for x in np.arange(int(env.observation_space.low[0]), int(env.observation_space.high[0])):
        for y in np.arange(int(env.observation_space.low[1]), int(env.observation_space.high[1])):
            if -2 <= x <= 6 and 2 <= y <= 6:
                continue
            pos = (x, y)
            #ori = 0
            for ori in np.arange(-3.14, 3.14, step=0.3):
                state = np.array((x, y, ori))
                action, _ = cec.select_action(env.normalize(state), evaluate=True)
                plot_loc = np.array(pos) + 0.5
                direction_x = np.cos(action[1] + ori) * (action[0])
                direction_y = np.sin(action[1] + ori) * (action[0])
                arrow = Arrow(plot_loc[0], plot_loc[1], direction_x, direction_y, width=0.05, color='k')
                ax.add_patch(arrow)

    return fig

def plot_trace_actions(cec, env):

    fig, ax = plt.subplots()
    width = (env.observation_space.low[0], env.observation_space.high[0])
    height = (env.observation_space.low[1], env.observation_space.high[1])
    ax.set_xlim(width)
    ax.set_ylim(height)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for x in np.arange(int(env.observation_space.low[0]), int(env.observation_space.high[0])):
        for y in np.arange(int(env.observation_space.low[1]), int(env.observation_space.high[1])):
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='white'))
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))
    ax.axvline(width[0],0,height[1],linewidth=5,c='k')
    ax.axvline(width[1],0,height[1],linewidth=5,c='k')
    ax.axhline(width[0],0,width[1],linewidth=5,c='k')
    ax.axhline(height[1],0,width[1],linewidth=5,c='k')
    done = False
    s = env.reset()
    while not done:
        action, _ = cec.select_action(s, evaluate=True)
        n_s, r, done, _ = env.step(action)
        pos = s[:2]
        direction_x = np.cos(action[1] + env.env.get_ori()) * (action[0])
        direction_y = np.sin(action[1] + env.env.get_ori()) * (action[0])

        plot_loc = np.array(pos) + 0.5
        arrow = Arrow(plot_loc[0], plot_loc[1], direction_x, direction_y, width=0.05, color='k')
        ax.add_patch(arrow)
        
        s = n_s

    return fig

def plot_retuns(cec, env):

    fig, ax = plt.subplots()
    width = (env.observation_space.low[0], env.observation_space.high[0])
    height = (env.observation_space.low[1], env.observation_space.high[1])
    ax.set_xlim(width)
    ax.set_ylim(height)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for x in np.arange(int(env.observation_space.low[0]), int(env.observation_space.high[0])):
        for y in np.arange(int(env.observation_space.low[1]), int(env.observation_space.high[1])):
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='white'))
            ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))
    ax.axvline(width[0],0,height[1],linewidth=5,c='k')
    ax.axvline(width[1],0,height[1],linewidth=5,c='k')
    ax.axhline(width[0],0,width[1],linewidth=5,c='k')
    ax.axhline(height[1],0,width[1],linewidth=5,c='k')
    for x in np.arange(int(env.observation_space.low[0]), int(env.observation_space.high[0])):
        for y in np.arange(int(env.observation_space.low[1]), int(env.observation_space.high[1])):
            if -2 <= x <= 6 and 2 <= y <= 6:
                continue
            pos = (x, y)
            #ori = 0
            for ori in np.arange(-3.14, 3.14, step=0.3):
                state = np.array((x, y, ori))
                value, _ = cec.get_action_value(env.normalize(state))
                plot_loc = np.array(pos) + 0.5
                direction_x = np.cos(ori) * value
                direction_y = np.sin(ori) * value
                arrow = Arrow(plot_loc[0], plot_loc[1], direction_x, direction_y, width=0.05, color='k')
                ax.add_patch(arrow)

    return fig

