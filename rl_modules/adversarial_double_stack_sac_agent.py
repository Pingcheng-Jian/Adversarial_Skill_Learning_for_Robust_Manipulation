import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads, sync_grads_for_tensor
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor_prob, actor, critic
from rl_modules.simhash import HashingBonusEvaluator
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from torch.nn.utils import clip_grad_norm_

"""
double agent (sac)
"""

class adversarial_double_stack_sac_agent:
    def __init__(self, args, env, env_params, writer=None):
        if args.cuda:
            torch.cuda.set_device(args.device)
        self.args = args
        self.env = env
        env_params['action'] = env_params['action'] // 2
        env_params['obs'] = 37  # original 56
        self.env_params = env_params
        # create the network
        self.actor_network_1 = actor_prob(env_params)
        self.critic_network_1 = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network_1)
        sync_networks(self.critic_network_1)
        # build up the target network
        self.actor_target_network_1 = actor_prob(env_params)
        self.critic_target_network_1 = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network_1.load_state_dict(self.actor_network_1.state_dict())
        self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())

        # create the network
        self.actor_network_2 = actor_prob(env_params)
        self.critic_network_2 = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network_2)
        sync_networks(self.critic_network_2)
        # build up the target network
        self.actor_target_network_2 = actor_prob(env_params)
        self.critic_target_network_2 = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network_2.load_state_dict(self.actor_network_2.state_dict())
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network_1.to(args.device)
            self.critic_network_1.to(args.device)
            self.actor_target_network_1.to(args.device)
            self.critic_target_network_1.to(args.device)
            self.actor_network_2.to(args.device)
            self.critic_network_2.to(args.device)
            self.actor_target_network_2.to(args.device)
            self.critic_target_network_2.to(args.device)
        # create the optimizer
        self.actor_optim_1 = torch.optim.Adam(self.actor_network_1.parameters(), lr=self.args.lr_actor)
        self.critic_optim_1 = torch.optim.Adam(self.critic_network_1.parameters(), lr=self.args.lr_critic)
        self.actor_optim_2 = torch.optim.Adam(self.actor_network_2.parameters(), lr=self.args.lr_actor_2)
        self.critic_optim_2 = torch.optim.Adam(self.critic_network_2.parameters(), lr=self.args.lr_critic_2)
        # her sampler
        if 'Stack' in self.args.env_name or 'Lift' in self.args.env_name:
            self.her_module_1 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward,
                                            stack=True)
            self.her_module_2 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward,
                                            stack=True)
        else:
            self.her_module_1 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
            self.her_module_2 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)

        # if args.only_for_door:
        #     self.her_module_2 = her_sampler(self.args.replay_strategy, self.args.replay_k,
        #                                     self.env.compute_reward_only_for_door, double=True)
        # else:
        #     self.her_module_2 = her_sampler(self.args.replay_strategy, self.args.replay_k,
        #                                     self.env.compute_reward_for_door, double=True)

        # create the replay buffer
        self.buffer_1 = replay_buffer(self.env_params, self.args.buffer_size, self.her_module_1.sample_her_transitions)
        self.buffer_2 = replay_buffer(self.env_params, self.args.buffer_size, self.her_module_2.sample_her_transitions)
        # self.buffer_2 = replay_buffer(self.env_params, self.args.buffer_size, self.her_module_2.sample_her_transitions,
        #                               double=True)

        # create the normalizer
        self.o_norm_1 = normalizer(size=37, default_clip_range=self.args.clip_range)  # original 56
        self.o_norm_2 = normalizer(size=37, default_clip_range=self.args.clip_range)  # original 56
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # for visualization
        self.g_norm_1 = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        self.g_norm_2 = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        self.explore = None
        if args.count_exp:
            self.explore = HashingBonusEvaluator(obs_processed_flat_dim=37,
                                                 beta=args.exp_beta)  # original 56
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            if args.count_exp:
                self.model_path = os.path.join(self.model_path, 'count-exploration')
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            # flag = 'actor_lr_' + str(self.args.lr_actor) + '_critic_lr_' + str(
            #     self.args.lr_critic) + '_actor_lr_2_' + str(self.args.lr_actor_2) + '_critic_lr_2_' + str(
            #     self.args.lr_critic_2)
            # flag += '_shaped_reward'
            # if args.only_for_door:
            #     flag += '_only_for_door'
            # else:
            #     flag += '_not_only_for_door'
            flag = str(self.args.env_name)
            self.model_path = os.path.join(self.model_path, flag)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            if not os.path.exists(self.args.save_training_success_rate_dir):
                os.mkdir(self.args.save_training_success_rate_dir)
            self.training_success_rate_path = os.path.join(self.args.save_training_success_rate_dir, self.args.env_name)
            if not os.path.exists(self.training_success_rate_path):
                os.mkdir(self.training_success_rate_path)
            self.training_success_rate_path = os.path.join(self.training_success_rate_path, flag)
            if not os.path.exists(self.training_success_rate_path):
                os.mkdir(self.training_success_rate_path)

            if not os.path.exists(self.args.save_training_return_dir):
                os.mkdir(self.args.save_training_return_dir)
            self.training_return_path = os.path.join(self.args.save_training_return_dir, self.args.env_name)
            if not os.path.exists(self.training_return_path):
                os.mkdir(self.training_return_path)
            self.training_return_path = os.path.join(self.training_return_path, flag)
            if not os.path.exists(self.training_return_path):
                os.mkdir(self.training_return_path)

        self.writer = writer
        # for sac, one more critic for each
        self.critic_network_1_2 = critic(env_params)
        self.critic_network_2_2 = critic(env_params)
        sync_networks(self.critic_network_1_2)
        sync_networks(self.critic_network_2_2)
        self.critic_target_network_1_2 = critic(env_params)
        self.critic_target_network_2_2 = critic(env_params)
        # load the weights into the target networks
        self.critic_target_network_1_2.load_state_dict(self.critic_network_1_2.state_dict())
        self.critic_target_network_2_2.load_state_dict(self.critic_network_2_2.state_dict())
        # if use gpu
        if self.args.cuda:
            self.critic_network_1_2.to(args.device)
            self.critic_target_network_1_2.to(args.device)
            self.critic_network_2_2.to(args.device)
            self.critic_target_network_2_2.to(args.device)
        # create the optimizer
        self.critic_optim_1_2 = torch.optim.Adam(self.critic_network_1_2.parameters(), lr=self.args.lr_critic)
        self.critic_optim_2_2 = torch.optim.Adam(self.critic_network_2_2.parameters(), lr=self.args.lr_critic_2)
        self.alpha_1 = args.alpha  # 0.2
        self.alpha_2 = args.alpha  # 0.2
        single_action_space = (env.action_space.shape[0] // 2,)
        if self.args.cuda:
            self.target_entropy_1 = -torch.prod(torch.FloatTensor(single_action_space).to(self.args.device)).item()
        if self.args.cuda:
            self.log_alpha_1 = torch.zeros(1, requires_grad=True, device=self.args.device)
            self.alpha_optim_1 = torch.optim.Adam([self.log_alpha_1], lr=args.alpha_lr)  # two agent share same alpha_lr now
        if self.args.cuda:
            self.target_entropy_2 = -torch.prod(torch.FloatTensor(single_action_space).to(self.args.device)).item()
        if self.args.cuda:
            self.log_alpha_2 = torch.zeros(1, requires_grad=True, device=self.args.device)
            self.alpha_optim_2 = torch.optim.Adam([self.log_alpha_2], lr=args.alpha_lr)  # two agent share same alpha_lr now

    def learn(self):
        """
        train the network
        """
        return_record = []
        success_rate_record = []
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs_1, mb_obs_2, mb_ag, mb_g, mb_actions_1, mb_actions_2, mb_door_dist, mb_time = [], [], [], [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs_1, ep_obs_2, ep_ag, ep_g, ep_actions_1, ep_actions_2, ep_door_dist = [], [], [], [], [], [], []  # ag means achieved goal
                    # reset the environment
                    observation = self.env.reset()
                    # print("low:")
                    # print(self.env.action_space.low)  # -1 8-D
                    # print("high:")
                    # print(self.env.action_space.high)  # 1 8-D
                    # print(observation)
                    obs_1 = observation['obs_1']
                    obs_2 = observation['obs_2']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor_1 = self._preproc_inputs(obs_1, g, 1)
                            input_tensor_2 = self._preproc_inputs(obs_2, g, 2)
                            pi_1, _, _ = self.actor_network_1.sample(input_tensor_1)
                            pi_2, _, _ = self.actor_network_2.sample(input_tensor_2)
                            action_1 = self._select_actions(pi_1)
                            action_2 = self._select_actions(pi_2)
                            # action_2 = np.zeros(4)
                            action = np.concatenate([action_1, action_2], -1)
                        # feed the actions into the environment
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                        observation_new, _, _, info = self.env.step(action)
                        obs_new_1 = observation_new['obs_1']
                        obs_new_2 = observation_new['obs_2']
                        ag_new = observation_new['achieved_goal']
                        dg_new = observation_new['desired_goal']
                        # ep_door_dist.append(info['door_distance'].copy())
                        # append rollouts
                        ep_obs_1.append(obs_1.copy())
                        ep_obs_2.append(obs_2.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions_1.append(action_1.copy())
                        ep_actions_2.append(action_2.copy())
                        # re-assign the observation
                        obs_1 = obs_new_1
                        obs_2 = obs_new_2
                        ag = ag_new
                        if self.args.count_exp:
                            self.explore.inc_hash(np.expand_dims(obs_2, 0))
                    ep_obs_1.append(obs_1.copy())
                    ep_obs_2.append(obs_2.copy())
                    ep_ag.append(ag.copy())
                    # ep_door_dist.append(info['door_distance'].copy())
                    mb_obs_1.append(ep_obs_1)
                    mb_obs_2.append(ep_obs_2)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions_1.append(ep_actions_1)
                    mb_actions_2.append(ep_actions_2)
                    # mb_door_dist.append(ep_door_dist)
                    mb_time.append(np.arange(self.env_params['max_timesteps']))
                # convert them into arrays
                mb_obs_1 = np.array(mb_obs_1)
                mb_obs_2 = np.array(mb_obs_2)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions_1 = np.array(mb_actions_1)
                mb_actions_2 = np.array(mb_actions_2)
                # mb_door_dist = np.array(mb_door_dist)
                mb_time = np.array(mb_time)
                # store the episodes
                self.buffer_1.store_episode([mb_obs_1, mb_ag, mb_g, mb_actions_1, mb_time])
                self.buffer_2.store_episode([mb_obs_2, mb_ag, mb_g, mb_actions_2, mb_time])
                # self.buffer_2.store_episode([mb_obs_2, mb_ag, mb_g, mb_actions_2, mb_door_dist])
                self._update_normalizer([mb_obs_1, mb_ag, mb_g, mb_actions_1], idx=1)
                self._update_normalizer([mb_obs_2, mb_ag, mb_g, mb_actions_2], idx=2)
                # self._update_normalizer([mb_obs_2, mb_ag, mb_g, mb_actions_2, mb_door_dist], idx=2)
                for nb in range(self.args.n_batches):
                    # train the network
                    if nb == self.args.n_batches - 1:
                        self._update_network(epoch, True)
                    else:
                        self._update_network(epoch)
                # soft update
                self._soft_update_target_network(self.actor_target_network_1, self.actor_network_1)
                self._soft_update_target_network(self.critic_target_network_1, self.critic_network_1)
                self._soft_update_target_network(self.critic_target_network_1_2, self.critic_network_1_2)
                self._soft_update_target_network(self.actor_target_network_2, self.actor_network_2)
                self._soft_update_target_network(self.critic_target_network_2, self.critic_network_2)
                self._soft_update_target_network(self.critic_target_network_2_2, self.critic_network_2_2)
            # start to do the evaluation
            # success_rate = self._eval_agent()
            success_rate, mean_reward = self._eval_agent_ver2()
            success_rate_record.append(success_rate)
            return_record.append(mean_reward)
            np.save(self.training_success_rate_path + '/adv_sac_double_pick_sr_seed' + str(
                self.args.seed) + '.npy',
                    np.array(success_rate_record))
            np.save(self.training_return_path + '/adv_sac_double_pick_return_seed' + str(
                self.args.seed) + '.npy',
                    np.array(return_record))
            if self.writer is not None:
                self.writer.add_scalar('success_rate', success_rate, epoch)
                self.writer.add_scalar('mean_reward', mean_reward, epoch)

            if self.args.seed == 1:
                # print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                print('epoch is: {}, eval success rate is: {:.3f}, mean reward is: {:.3f}'.format(epoch, success_rate,
                                                                                                  mean_reward))
                torch.save(
                    [self.o_norm_1.mean, self.o_norm_1.std, self.o_norm_2.mean, self.o_norm_2.std, self.g_norm.mean,
                     self.g_norm.std,
                     self.actor_network_1.state_dict(), self.actor_network_2.state_dict()], \
                    self.model_path + '/adv_sac_double_pick.pt')
        return success_rate_record, return_record
        # o_norm and g_norm are defined in line 48 49. The obs and goal normalizer.

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, idx):
        if idx == 1:
            obs_norm = self.o_norm_1.normalize(obs)
        elif idx == 2:
            obs_norm = self.o_norm_2.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.args.device)
        return inputs

    def _preproc_inputs_1(self, obs, g, idx):
        if idx == 1:
            obs_norm = self.o_norm_1.normalize(obs)
        elif idx == 2:
            obs_norm = self.o_norm_2.normalize(obs)
        g_norm = self.g_norm_1.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.args.device)
        return inputs

    def _preproc_inputs_2(self, obs, g, idx):
        if idx == 1:
            obs_norm = self.o_norm_1.normalize(obs)
        elif idx == 2:
            obs_norm = self.o_norm_2.normalize(obs)
        g_norm = self.g_norm_2.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.to(self.args.device)
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # print(self.env_params['action_max'])  # print and check the action_max = 1.0 for reach and push
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch, idx):
        if idx == 1:
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        elif idx == 2:
            # mb_obs, mb_ag, mb_g, mb_actions, mb_door_dist = episode_batch
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]  # 50
        # create the new buffer to store them

        if idx == 1:
            buffer_temp = {'obs': mb_obs,  # (2, 51, 45)
                           'ag': mb_ag,  # (2, 51, 3)
                           'g': mb_g,  # (2, 50, 3)
                           'actions': mb_actions,  # (2, 50 ,8)
                           'obs_next': mb_obs_next,  # (2, 50, 45)
                           'ag_next': mb_ag_next,  # (2, 50, 3)
                           }
            transitions = self.her_module_1.sample_her_transitions(buffer_temp, num_transitions)
        elif idx == 2:
            buffer_temp = {'obs': mb_obs,  # (2, 51, 45)
                           'ag': mb_ag,  # (2, 51, 3)
                           'g': mb_g,  # (2, 50, 3)
                           'actions': mb_actions,  # (2, 50 ,8)
                           'obs_next': mb_obs_next,  # (2, 50, 45)
                           'ag_next': mb_ag_next,  # (2, 50, 3)
                           # 'door_distance': mb_door_dist
                           }
            transitions = self.her_module_2.sample_her_transitions(buffer_temp, num_transitions)
        else:
            raise NotImplementedError

        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        if idx == 1:
            self.o_norm_1.update(transitions['obs'])
        elif idx == 2:
            self.o_norm_2.update(transitions['obs'])

        self.g_norm.update(transitions['g'])
        # recompute the stats
        if idx == 1:
            self.o_norm_1.recompute_stats()
        elif idx == 2:
            self.o_norm_2.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)  # limited between -200 to 200
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, step, if_write=False):
        # # for agent 1, the protagonist agent

        # sample the episodes
        transitions = self.buffer_1.sample(self.args.batch_size)  # sample from the replay_buffer
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm_1.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm_1.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.to(self.args.device)
            inputs_next_norm_tensor = inputs_next_norm_tensor.to(self.args.device)
            actions_tensor = actions_tensor.to(self.args.device)
            r_tensor = r_tensor.to(self.args.device)
        # calculate the target Q value function
        with torch.no_grad():  # wrapped by this, the grad_fn don't track this part
            # do the normalization
            # concatenate the stuffs
            actions_next, next_state_logp, _ = self.actor_target_network_1.sample(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network_1(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            q_next_value_1_2 = self.critic_target_network_1_2(inputs_next_norm_tensor, actions_next)
            q_next_value_1_2 = q_next_value_1_2.detach()
            target_q_value_1_2 = r_tensor + self.args.gamma * q_next_value_1_2
            target_q_value_1_2 = target_q_value_1_2.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            # make target_q_value between -clip_return and 0
            target_q_value = torch.min(target_q_value, target_q_value_1_2) - self.alpha_1 * next_state_logp
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network_1(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the q loss
        real_q_value_1_2 = self.critic_network_1_2(inputs_norm_tensor, actions_tensor)
        critic_loss_1_2 = (target_q_value_1_2 - real_q_value_1_2).pow(2).mean()
        # the actor loss
        actions_real, logp, _ = self.actor_network_1.sample(inputs_norm_tensor)
        actor_loss = (self.alpha_1 * logp - torch.min(
            self.critic_network_1(inputs_norm_tensor, actions_real),
            self.critic_network_1_2(inputs_norm_tensor, actions_real))).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        alpha_loss_1 = -(self.log_alpha_1 * (logp + self.target_entropy_1).detach()).mean()

        self.alpha_optim_1.zero_grad()
        alpha_loss_1.backward()
        sync_grads_for_tensor(self.log_alpha_1)
        self.alpha_optim_1.step()
        self.alpha_1 = self.log_alpha_1.exp()

        # start to update the network
        self.actor_optim_1.zero_grad()
        actor_loss.backward()

        # # clip the grad
        # clip_grad_norm_(self.actor_network_1.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.actor_network_1)
        self.actor_optim_1.step()
        # update the critic_network
        self.critic_optim_1.zero_grad()
        critic_loss.backward()

        # # clip the grad
        # clip_grad_norm_(self.critic_network_1.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.critic_network_1)
        self.critic_optim_1.step()

        self.critic_optim_1_2.zero_grad()
        critic_loss_1_2.backward()

        # # clip the grad
        # clip_grad_norm_(self.critic_network_1_2.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.critic_network_1_2)
        self.critic_optim_1_2.step()
        if self.writer is not None and if_write:
            self.writer.add_scalar('actor_loss_1', actor_loss, step)
            self.writer.add_scalar('critic_loss_1', critic_loss, step)

        """"""
        # # for agent 2, the adversary agent

        # sample the episodes
        transitions = self.buffer_2.sample(self.args.batch_size)  # sample from the replay_buffer
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm_2.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm_2.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.to(self.args.device)
            inputs_next_norm_tensor = inputs_next_norm_tensor.to(self.args.device)
            actions_tensor = actions_tensor.to(self.args.device)
            r_tensor = r_tensor.to(self.args.device)
        # calculate the target Q value function
        with torch.no_grad():  # wrapped by this, the grad_fn don't track this part
            # do the normalization
            # concatenate the stuffs
            actions_next, next_state_logp, _ = self.actor_target_network_2.sample(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network_2(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            q_next_value_2_2 = self.critic_target_network_2_2(inputs_next_norm_tensor, actions_next)
            q_next_value_2_2 = q_next_value_2_2.detach()
            target_q_value_2_2 = r_tensor + self.args.gamma * q_next_value_2_2
            target_q_value_2_2 = target_q_value_2_2.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            # make target_q_value between -clip_return and 0
            target_q_value = torch.min(target_q_value, target_q_value_2_2) - self.alpha_2 * next_state_logp
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network_2(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the q loss
        real_q_value_2_2 = self.critic_network_2_2(inputs_norm_tensor, actions_tensor)
        critic_loss_2_2 = (target_q_value_2_2 - real_q_value_2_2).pow(2).mean()
        # the actor loss
        actions_real, logp, _ = self.actor_network_2.sample(inputs_norm_tensor)
        actor_loss = (self.alpha_2 * logp + torch.min(
            self.critic_network_2(inputs_norm_tensor, actions_real),
            self.critic_network_2_2(inputs_norm_tensor, actions_real))).mean()  # for adversary agent
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        alpha_loss_2 = -(self.log_alpha_2 * (logp + self.target_entropy_2).detach()).mean()


        self.alpha_optim_2.zero_grad()
        alpha_loss_2.backward()
        sync_grads_for_tensor(self.log_alpha_2)
        self.alpha_optim_2.step()
        self.alpha_2 = self.log_alpha_2.exp()

        # start to update the network
        self.actor_optim_2.zero_grad()
        actor_loss.backward()

        # # clip grad loss
        # clip_grad_norm_(self.actor_network_2.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.actor_network_2)
        self.actor_optim_2.step()
        # update the critic_network
        self.critic_optim_2.zero_grad()
        critic_loss.backward()

        # # clip grad loss
        # clip_grad_norm_(self.critic_network_2.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.critic_network_2)
        self.critic_optim_2.step()

        self.critic_optim_2_2.zero_grad()
        critic_loss_2_2.backward()

        # # clip grad loss
        # clip_grad_norm_(self.critic_network_2_2.parameters(), max_norm=20, norm_type=2)

        sync_grads(self.critic_network_2_2)
        self.critic_optim_2_2.step()
        if self.writer is not None and if_write:
            self.writer.add_scalar('actor_loss_2', actor_loss, step)
            self.writer.add_scalar('critic_loss_2', critic_loss, step)

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi_1 = self.actor_network_1(input_tensor)
                    pi_2 = self.actor_network_2(input_tensor)
                    # convert the actions
                    pi = torch.cat([pi_1, pi_2], -1)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def _eval_agent_ver2(self):
        total_success_rate = []
        reward_list = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs_1 = observation['obs_1']
            obs_2 = observation['obs_2']
            g = observation['desired_goal']
            reward_sum = 0
            for _ in range(self.args.max_episode_steps):
                with torch.no_grad():
                    input_tensor_1 = self._preproc_inputs(obs_1, g, 1)
                    input_tensor_2 = self._preproc_inputs(obs_2, g, 2)
                    pi_1, _, _ = self.actor_network_1.sample(input_tensor_1)
                    pi_2, _, _ = self.actor_network_2.sample(input_tensor_2)
                    # convert the actions
                    # pi_2 = torch.zeros_like(pi_1)
                    pi = torch.cat([pi_1, pi_2], -1)
                    actions = pi.detach().cpu().numpy().squeeze()
                actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
                observation_new, reward, _, info = self.env.step(actions)  # get info and reward
                reward_sum = reward_sum + reward
                obs_1 = observation_new['obs_1']
                obs_2 = observation_new['obs_2']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            reward_list.append(reward_sum)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)

        reward_list = np.array(reward_list)
        local_reward = np.mean(reward_list)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)
        return (global_success_rate / MPI.COMM_WORLD.Get_size()), (global_reward / MPI.COMM_WORLD.Get_size())

    def visualize(self, model_path):
        # agent 2 is adversary agent
        total_success_rate = []
        data = torch.load(model_path, map_location=torch.device('cpu'))
        self.o_norm_1.mean = data[0]
        self.o_norm_1.std = data[1]
        self.o_norm_2.mean = data[2]
        self.o_norm_2.std = data[3]
        self.g_norm.mean = data[4]
        self.g_norm.std = data[5]
        self.actor_network_1.load_state_dict(data[-2])
        self.actor_network_1.eval()
        self.actor_network_2.load_state_dict(data[-1])
        self.actor_network_2.eval()

        for _ in range(20):
            per_success_rate = []
            observation = self.env.reset()
            obs_1 = observation['obs_1']
            obs_2 = observation['obs_2']
            g = observation['desired_goal']
            for t in range(self.args.max_episode_steps):
                with torch.no_grad():
                    input_tensor_1 = self._preproc_inputs(obs_1, g, 1)
                    input_tensor_2 = self._preproc_inputs(obs_2, g, 2)
                    pi_1 = self.actor_network_1(input_tensor_1)
                    pi_2 = self.actor_network_2(input_tensor_2)
                    pi_1 = pi_1[0]
                    pi_2 = pi_2[0]
                    pi_2 = pi_2 * self.args.test_ad_factor
                    # print(pi_1)
                    # convert the actions
                    # pi_1 = torch.zeros_like(pi_2)
                    pi = torch.cat([pi_1, pi_2], -1)
                    action = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action)
                self.env.render()
                obs_1 = observation_new['obs_1']
                obs_2 = observation_new['obs_2']
                per_success_rate.append(info['is_success'])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        print(local_success_rate)

    def visualize_normal_ad(self, model_path, normal_model_path):
        # agent 2 is adversary agent
        # total_success_rate = []
        data = torch.load(model_path, map_location=torch.device('cpu'))
        normal_data = torch.load(normal_model_path, map_location=torch.device('cpu'))
        self.o_norm_1.mean = normal_data[0]
        self.o_norm_1.std = normal_data[1]
        self.o_norm_2.mean = data[2]
        self.o_norm_2.std = data[3]
        self.g_norm_1.mean = normal_data[4]
        self.g_norm_1.std = normal_data[5]
        self.g_norm_2.mean = data[4]
        self.g_norm_2.std = data[5]
        self.actor_network_1.load_state_dict(normal_data[-2])
        self.actor_network_1.eval()
        self.actor_network_2.load_state_dict(data[-1])
        self.actor_network_2.eval()

        success_rate_list = []

        for ite in range(10):
            total_success_rate = []
            self.args.seed = 121

            for _ in range(20):
                per_success_rate = []
                observation = self.env.reset()
                obs_1 = observation['obs_1']
                obs_2 = observation['obs_2']
                g = observation['desired_goal']
                for t in range(self.args.max_episode_steps):
                    with torch.no_grad():
                        input_tensor_1 = self._preproc_inputs_1(obs_1, g, 1)
                        input_tensor_2 = self._preproc_inputs_2(obs_2, g, 2)
                        pi_1 = self.actor_network_1(input_tensor_1)
                        pi_2 = self.actor_network_2(input_tensor_2)
                        pi_1 = pi_1[0]
                        pi_2 = pi_2[0]
                        pi_2 = pi_2 * self.args.test_ad_factor
                        # pi_2 = torch.zeros_like(pi_1)
                        # print(pi_1)
                        # convert the actions
                        # pi_1 = torch.zeros_like(pi_2)
                        pi = torch.cat([pi_1, pi_2], -1)
                        action = pi.detach().cpu().numpy().squeeze()
                    observation_new, _, _, info = self.env.step(action)
                    self.env.render()
                    obs_1 = observation_new['obs_1']
                    obs_2 = observation_new['obs_2']
                    per_success_rate.append(info['is_success'])

                total_success_rate.append(per_success_rate)

            total_success_rate = np.array(total_success_rate)
            local_success_rate = np.mean(total_success_rate[:, -1])
            success_rate_list.append(local_success_rate)
            # print(local_success_rate)

        mean_sr = np.mean(np.array(success_rate_list))
        std_sr = np.std(np.array(success_rate_list))
        print(success_rate_list)
        print(mean_sr)
        print(std_sr)

        # for _ in range(20):
        #     per_success_rate = []
        #     observation = self.env.reset()
        #     obs_1 = observation['obs_1']
        #     obs_2 = observation['obs_2']
        #     g = observation['desired_goal']
        #     for t in range(self.args.max_episode_steps):
        #         with torch.no_grad():
        #             input_tensor_1 = self._preproc_inputs_1(obs_1, g, 1)
        #             input_tensor_2 = self._preproc_inputs_2(obs_2, g, 2)
        #             pi_1 = self.actor_network_1(input_tensor_1)
        #             pi_2 = self.actor_network_2(input_tensor_2)
        #             pi_1 = pi_1[0]
        #             pi_2 = pi_2[0]
        #             pi_2 = pi_2 * self.args.test_ad_factor
        #             # pi_2 = torch.zeros_like(pi_1)
        #             # print(pi_1)
        #             # convert the actions
        #             # pi_1 = torch.zeros_like(pi_2)
        #             pi = torch.cat([pi_1, pi_2], -1)
        #             action = pi.detach().cpu().numpy().squeeze()
        #         observation_new, _, _, info = self.env.step(action)
        #         # self.env.render()
        #         obs_1 = observation_new['obs_1']
        #         obs_2 = observation_new['obs_2']
        #         per_success_rate.append(info['is_success'])
        #
        #     total_success_rate.append(per_success_rate)
        #
        # total_success_rate = np.array(total_success_rate)
        # local_success_rate = np.mean(total_success_rate[:, -1])
        # print(local_success_rate)

    def visualize_random_noise_test(self, normal_model_path):
        # agent 2 is adversary agent
        # total_success_rate = []
        normal_data = torch.load(normal_model_path, map_location=torch.device('cpu'))
        self.o_norm_1.mean = normal_data[0]
        self.o_norm_1.std = normal_data[1]
        self.g_norm_1.mean = normal_data[4]
        self.g_norm_1.std = normal_data[5]
        self.actor_network_1.load_state_dict(normal_data[-2])
        self.actor_network_1.eval()

        success_rate_list = []

        for ite in range(10):
            total_success_rate = []
            self.args.seed = 121+ite

            for _ in range(20):
                per_success_rate = []
                observation = self.env.reset()
                obs_1 = observation['obs_1']
                obs_2 = observation['obs_2']
                g = observation['desired_goal']
                for t in range(self.args.max_episode_steps):
                    with torch.no_grad():
                        input_tensor_1 = self._preproc_inputs_1(obs_1, g, 1)
                        input_tensor_2 = self._preproc_inputs_2(obs_2, g, 2)
                        pi_1 = self.actor_network_1(input_tensor_1)
                        # pi_2 = self.actor_network_2(input_tensor_2)
                        pi_1 = pi_1[0]
                        # pi_2 = pi_2[0]
                        pi_2 = np.random.uniform(-1, 1, 4)
                        pi_2 = np.atleast_2d(pi_2)
                        pi_2 = torch.from_numpy(pi_2)
                        pi_2 = pi_2 * self.args.test_ad_factor
                        pi = torch.cat([pi_1, pi_2], -1)
                        action = pi.detach().cpu().numpy().squeeze()
                    observation_new, _, _, info = self.env.step(action)
                    self.env.render()
                    obs_1 = observation_new['obs_1']
                    obs_2 = observation_new['obs_2']
                    per_success_rate.append(info['is_success'])

                total_success_rate.append(per_success_rate)

            total_success_rate = np.array(total_success_rate)
            local_success_rate = np.mean(total_success_rate[:, -1])
            success_rate_list.append(local_success_rate)
            # print(local_success_rate)

        mean_sr = np.mean(np.array(success_rate_list))
        std_sr = np.std(np.array(success_rate_list))
        print(success_rate_list)
        print(mean_sr)
        print(std_sr)

        # for _ in range(20):
        #     per_success_rate = []
        #     observation = self.env.reset()
        #     obs_1 = observation['obs_1']
        #     obs_2 = observation['obs_2']
        #     g = observation['desired_goal']
        #     for t in range(self.args.max_episode_steps):
        #         with torch.no_grad():
        #             input_tensor_1 = self._preproc_inputs_1(obs_1, g, 1)
        #             input_tensor_2 = self._preproc_inputs_2(obs_2, g, 2)
        #             pi_1 = self.actor_network_1(input_tensor_1)
        #             pi_2 = self.actor_network_2(input_tensor_2)
        #             pi_1 = pi_1[0]
        #             pi_2 = pi_2[0]
        #             pi_2 = pi_2 * self.args.test_ad_factor
        #             # pi_2 = torch.zeros_like(pi_1)
        #             # print(pi_1)
        #             # convert the actions
        #             # pi_1 = torch.zeros_like(pi_2)
        #             pi = torch.cat([pi_1, pi_2], -1)
        #             action = pi.detach().cpu().numpy().squeeze()
        #         observation_new, _, _, info = self.env.step(action)
        #         # self.env.render()
        #         obs_1 = observation_new['obs_1']
        #         obs_2 = observation_new['obs_2']
        #         per_success_rate.append(info['is_success'])
        #
        #     total_success_rate.append(per_success_rate)
        #
        # total_success_rate = np.array(total_success_rate)
        # local_success_rate = np.mean(total_success_rate[:, -1])
        # print(local_success_rate)
