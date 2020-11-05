import numpy as np
from gym.envs.robotics import fetch_env


class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, double=False, Yunfei=False,
                 stack=False):  # replay_strategy = 'future'
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k  # replay_k = 4
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))  # 0.8
        else:
            self.future_p = 0
        self.reward_func = reward_func  # reward_func is the compute_reward function
        self.Yunfei = Yunfei  # bool value Yunfei is used for the environment developed by Yunfei Li
        self.double = double
        self.stack = stack
        if self.Yunfei:
            self.reward_func2 = fetch_env.FetchEnv.compute_reward
            self.reward_type = 'sparse'
            self.distance_threshold = 0.05

    # episode_batch = buffer_temp   batch_size_in_transitions = num_transitions
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]  # 50
        rollout_batch_size = episode_batch['actions'].shape[0]  # 2
        batch_size = batch_size_in_transitions  # 50
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        if self.double:
            transitions['r'] = np.expand_dims(
                self.reward_func(transitions['ag_next'], transitions['g'], None, transitions['door_distance']), 1)
        else:
            if self.Yunfei:
                transitions['r'] = np.expand_dims(
                    self.reward_func2(self, transitions['ag_next'], transitions['g'], None), 1)
            elif self.stack:
                transitions['r'] = np.expand_dims(self.reward_func(transitions['obs_next'], transitions['g'], None), 1)
            else:
                transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
