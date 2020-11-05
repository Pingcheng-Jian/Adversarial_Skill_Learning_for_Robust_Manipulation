import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""

LOG_SIG_MAX = 2
LOG_SIG_MIN = -9
epsilon = 1e-6


# define the actor network


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params, half=False):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        if half:
            self.action_out = nn.Linear(env_params['hidden_unit'], env_params['action'] // 2)
        else:
            self.action_out = nn.Linear(env_params['hidden_unit'], env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        a_out = self.action_out(x)
        actions = self.max_action * torch.tanh(a_out)

        return actions


class actor_prob(nn.Module):
    def __init__(self, env_params, half=False):
        super(actor_prob, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))
        # if half:
        #     self.mean_linear = init_(nn.Linear(env_params['hidden_unit'], env_params['action']//2))
        #     self.log_std_linear = init_(nn.Linear(env_params['hidden_unit'], env_params['action']//2))
        # else:
        #     self.mean_linear = init_(nn.Linear(env_params['hidden_unit'], env_params['action']))
        #     self.log_std_linear = init_(nn.Linear(env_params['hidden_unit'], env_params['action']))
        if half:
            self.mean_linear = nn.Linear(env_params['hidden_unit'], env_params['action'] // 2)
            self.log_std_linear = nn.Linear(env_params['hidden_unit'], env_params['action'] // 2)
        else:
            self.mean_linear = nn.Linear(env_params['hidden_unit'], env_params['action'])
            self.log_std_linear = nn.Linear(env_params['hidden_unit'], env_params['action'])

        self.action_scale = torch.tensor(env_params['action_max'])
        self.action_bias = torch.tensor(0.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        dist_ent = normal.entropy().mean()
        return action, log_prob, dist_ent

    def evaluation(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias) / self.action_scale
        x_t = torch.atanh(y_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob


class critic(nn.Module):
    def __init__(self, env_params, half=False):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        if half:
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] + 1,
                                 env_params['hidden_unit'])  # 1 is the idx of arm
        else:
            self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'],
                                 env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.q_out = nn.Linear(env_params['hidden_unit'], 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class value(nn.Module):
    def __init__(self, env_params):
        super(value, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.v_out = nn.Linear(env_params['hidden_unit'], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.v_out(x)
        return v


class dynamics(nn.Module):
    def __init__(self, env_params):
        super(dynamics, self).__init__()
        self.max_action = env_params['action_max']
        self.n_object = env_params['n_object']
        self.fc1 = nn.Linear(2 * env_params['grip'] + 3 * env_params['n_object'] + env_params['action'] + 4,
                             env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.out = nn.Linear(env_params['hidden_unit'], 2 * env_params['grip'] + 3 * env_params['n_object'] + 4)
        self._idx = np.concatenate(
            [np.arange(0, 3 * self.n_object),
             np.arange(9 * self.n_object, 9 * self.n_object + 3),
             np.arange(12 * self.n_object + 3, 12 * self.n_object + 5),
             np.arange(15 * self.n_object + 10, 15 * self.n_object + 13),
             np.arange(18 * self.n_object + 13, 18 * self.n_object + 15)])

    def forward(self, x, actions):
        x = x[:, self._idx]
        x_prev = x
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        diff = self.out(x)
        # diff = diff_norm.normalize_tensor(self.out(x), device=x.device)
        # pred = o_norm.normalize_tensor(diff + x_prev, device=x.device)
        return diff + x_prev

    def plan(self, x, actions):
        x_prev = x
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        diff = self.out(x)
        # diff = diff_norm.normalize_tensor(self.out(x), device=x.device)
        # pred = o_norm.normalize_tensor(diff + x_prev, device=x.device)
        return diff + x_prev

class reward(nn.Module):
    def __init__(self, env_params):
        super(reward, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.r_out = nn.Linear(env_params['hidden_unit'], 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        r = self.r_out(x)
        return r


class inverse_dynamics(nn.Module):
    def __init__(self, env_params):
        super(inverse_dynamics, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear((env_params['obs'] + env_params['goal']) * 2, env_params['hidden_unit'])
        self.fc2 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.fc3 = nn.Linear(env_params['hidden_unit'], env_params['hidden_unit'])
        self.out = nn.Linear(env_params['hidden_unit'], env_params['action'])

    def forward(self, x, x_next):
        x = torch.cat([x, x_next], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        a_out = self.out(x)
        actions = self.max_action * torch.tanh(a_out)
        return actions


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
