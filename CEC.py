import numpy
import numpy as np
from sklearn.neighbors import KDTree
#from Helper import softmax
#from scipy.special import softmax
from utils import softmax, ActionRescaler
from sklearn import preprocessing

# can be optimized by using separate list for each item and retrieve via index.
class CEC():
    def __init__(self, gamma, capacity, distance_threshold, action_space, eps_start=0.9, eps_end=0.05, eps_decay_steps=10000, exploration='random', act_noise=0.1, softmax_tau=1, k=5, T=1):
        self.tree = None
        self.states = []
        self.returns = []
        self.actions = []
        self.timesteps = []

        self.gamma = gamma
        self.capacity = capacity
        self.distance_threshold = distance_threshold
        self.global_cnt = 0
        self.action_space = action_space
        self.global_steps = 1
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.exploration = exploration
        self.act_noise = act_noise
        self.k = k
        self.softmax_tau = softmax_tau
        self.T = T
        self.action_scaler = ActionRescaler(action_high=action_space.high, action_low=action_space.low)
        #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

    def select_action(self, state, evaluate=False):
        self.epsilon = max(self.eps_start - ((self.eps_start - self.eps_end) / self.eps_decay_steps) * self.global_steps, self.eps_end)

        # always take the greedy action during the evaluation.
        if evaluate:
            existing_state_idx, distance = self.get_closest_state(state)
            if existing_state_idx is not None:
                action = self.actions[existing_state_idx]
                return action, distance
            elif existing_state_idx is None:
                action = self.action_space.sample()
                raise ValueError('This should not happen. You are trying to evaluating an empty agent.')
            
        else:
            # if evaluate is True, the function has already been returned.
            self.global_steps += 1
            knn_state_idx, knn_distance = self.get_knn_within_distance_threshold(state)
            if knn_state_idx is None:
                action = self.action_space.sample()
            else:
                # e-greedy policy
                if np.random.rand() < self.epsilon:
                    if self.exploration == 'random':
                        action = self.action_space.sample()
                    elif self.exploration == 'noise':
                        # take softmax over all returns of knns.
                        knn_returns = np.array(self.returns)[knn_state_idx]
                        knn_actions = np.array(self.actions)[knn_state_idx]
                        softmax_returns = softmax(knn_returns, tau=self.softmax_tau)
                        sample_idx = [m for m in range(len(knn_actions))]
                        action_idx = np.random.choice(sample_idx, p=softmax_returns)
                        action = knn_actions[action_idx]

                        # scale action to [-1, 1]
                        scaled_action = self.action_scaler.rescale_action(action)

                        # add noise to scaled action
                        noised_scaled_action = scaled_action + np.random.normal(scale=self.act_noise, size=scaled_action.shape)

                        # scale action back to original range
                        action = self.action_scaler.scale_back_action(noised_scaled_action)

                        # clip the action into normal action space.
                        action = action.clip(self.action_space.low, self.action_space.high)
                else:
                    # take softmax over all returns of knns.
                    knn_returns = np.array(self.returns)[knn_state_idx]
                    knn_actions = np.array(self.actions)[knn_state_idx]
                    softmax_returns = softmax(knn_returns, tau=self.softmax_tau)
                    sample_idx = [m for m in range(len(knn_actions))]
                    action_idx = np.random.choice(sample_idx, p=softmax_returns)
                    action = knn_actions[action_idx]
            
            return action

    def update(self, states, actions, rewards):
        T = len(actions)
        G = 0.0
        for t in reversed(range(T)):
            s = states[t]
            a = actions[t] 
            G = rewards[t] + self.gamma * G    
            c_state_idx, distance = self.get_closest_state(s)
            # add
            if distance > self.distance_threshold or c_state_idx is None:
                if len(self.returns) >= self.capacity:
                # if we need to add but the max capacity is already reached. First remove, then add.
                    self.remove_oldest()
                self.add(s, a, G, self.global_cnt)
                self.global_cnt += 1
            # update / replace
            elif distance <= self.distance_threshold:
                if self.returns[c_state_idx] < G:
                # if new return > old return, we replace it.
                    self.replace(c_state_idx, s, a, G, self.global_cnt)
                    self.global_cnt += 1
                else:
                    # only update the time step since old state/action is good enough.
                    self.timesteps[c_state_idx] = self.global_cnt
                    self.global_cnt += 1
            else:
                raise ValueError('check code, there is an unexpected error.')
        #print('Return: {}'.format(G))
        self.tree = KDTree(self.states)
        return G

    def add(self, s, a, G, ts):
        self.states.append(s)
        self.actions.append(a)
        self.returns.append(G)
        self.timesteps.append(ts)

    def replace(self, old_idx, n_s, n_a, n_G, n_ts):
        self.states[old_idx] = n_s
        self.returns[old_idx] = n_G
        self.actions[old_idx] = n_a
        self.timesteps[old_idx] = n_ts

    def remove_oldest(self):
        min_timestep_idx = int(np.argmin(self.timesteps))
        self.states.pop(min_timestep_idx)
        self.actions.pop(min_timestep_idx)
        self.returns.pop(min_timestep_idx)
        self.timesteps.pop(min_timestep_idx)

    def get_closest_state(self, state):
        if self.tree:
            distance, c_state_idx = self.tree.query([state], k=1)
            return c_state_idx[0][0], distance[0][0]
        else:
            return None, np.inf

    def get_knn_within_distance_threshold(self, state):
        if self.tree:
            size_tree = len(self.returns)
            if size_tree >= self.k:
                distance, c_state_idx = self.tree.query([state], k=self.k)
            else:
                distance, c_state_idx = self.tree.query([state], k=size_tree)
            # return a list of state idx and distances.
            c_state_idx = c_state_idx[distance < self.T * self.distance_threshold]
            c_state_idx = c_state_idx.ravel()
            distance = distance[distance < self.T * self.distance_threshold]
            distance = distance.ravel()
            if len(c_state_idx) == 0:
                return None, np.inf
            return c_state_idx, distance
        else:
            return None, np.inf

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
                "states": self.states,
                "returns": self.returns,
                "actions": self.actions,
                "timesteps": self.timesteps,
                })
        print("Memory module saved.")

    def get_action_value(self, state):
        # always take the greedy action during the evaluation.
        existing_state_idx, distance = self.get_closest_state(state)
        if existing_state_idx is not None:
            value = self.returns[existing_state_idx]
            return value, distance
        elif existing_state_idx is None:
            raise ValueError('This should not happen. You are trying to evaluating an empty agent.')
            
