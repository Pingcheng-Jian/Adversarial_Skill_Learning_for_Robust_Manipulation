import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, double=False, sil=False, reward_func=None, alt=False,
                 T=None, br=False, skill=False):
        self.env_params = env_params
        self.double = double
        self.sil = sil
        self.alt = alt
        self.br = br
        self.skill = skill
        if T is None:
            self.T = env_params['max_timesteps']
        else:
            self.T = T
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.reward_func = reward_func
        # create the buffer to store info
        if double:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            'door_distance': np.empty([self.size, self.T + 1, ])
                            }
        elif self.sil:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            't': np.empty([self.size, self.T, ]),
                            'R': np.empty([self.size, self.T, ]),
                            }
        elif self.br:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            'logp': np.empty([self.size, self.T, ]),
                            'stop': np.empty([self.size, self.T, 2]),
                            }
        elif self.skill:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            'grip_0': np.empty([self.size, self.T+1, self.env_params['grip']]),
                            'grip_1': np.empty([self.size, self.T+1, self.env_params['grip']]),
                            }
        else:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            't': np.empty([self.size, self.T, ]),
                            # 'R': np.empty([self.size, self.T, ]),
                            }

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        if self.double:
            mb_obs, mb_ag, mb_g, mb_actions, mb_door_dist = episode_batch
        elif self.sil:
            mb_obs, mb_ag, mb_g, mb_actions, mb_time, mb_R = episode_batch
        elif self.alt:
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        elif self.br:
            mb_obs, mb_ag, mb_g, mb_actions, mb_logp, mb_stop = episode_batch
        elif self.skill:
            mb_obs, mb_ag, mb_g, mb_actions, mb_grip0, mb_grip1 = episode_batch
        else:
            mb_obs, mb_ag, mb_g, mb_actions, mb_time = episode_batch

        batch_size = mb_obs.shape[0]

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            if not self.double:
                self.buffers['obs'][idxs] = mb_obs
                self.buffers['ag'][idxs] = mb_ag
                self.buffers['g'][idxs] = mb_g
                self.buffers['actions'][idxs] = mb_actions
                if self.br:
                    self.buffers['stop'][idxs] = mb_stop
                    self.buffers['logp'][idxs] = mb_logp
                elif self.skill:
                    self.buffers['grip_0'][idxs] = mb_grip0
                    self.buffers['grip_1'][idxs] = mb_grip1
                elif not self.alt:
                    self.buffers['t'][idxs] = mb_time
                if self.sil:
                    self.buffers['R'][idxs] = mb_R
            else:
                self.buffers['door_distance'][idxs] = mb_door_dist
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        if self.skill:
            temp_buffers['grip_next_0'] = temp_buffers['grip_0'][:, 1:, :]
            temp_buffers['grip_next_1'] = temp_buffers['grip_1'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def sample_wo_her(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        transitions = self.sample_sil_transitions(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def sample_sil_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
