import argparse
import datetime
import robel
import gym
import numpy as np
import itertools
import torch
from rl_modules.robel_sac import SAC
from rl_modules.robel_ad_sac import Adv_SAC
from torch.utils.tensorboard import SummaryWriter
from rl_modules.robel_replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="DClawTurnFixed-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=121, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

# for adversarial training
parser.add_argument('--po_iteration', type=int, default=100,
                    help='protagonist update 100 episodes')
parser.add_argument('--ad_iteration', type=int, default=100,
                    help='adversary update 100 episodes')
parser.add_argument('--ad_factor', type=float, default=0.6,
                    help='adversary action amplitude')

args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
po_agent = SAC(env.observation_space.shape[0], env.action_space, args)
ad_agent = Adv_SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/Adv_Train_{}_SAC_{}_{}_{}_{}_ad{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                     args.env_name,
                                                     args.policy,
                                                     "autotune" if args.automatic_entropy_tuning else "",
                                                     args.seed,
                                                     args.ad_factor))

# Memory
po_memory = ReplayMemory(args.replay_size, args.seed)
ad_memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
po_total_numsteps = 0
po_updates = 0
ad_total_numsteps = 0
ad_updates = 0

total_episode = 1
po_episode = 1
ad_episode = 1
for outer_iteration in itertools.count(1):

    for po_iteration in range(args.po_iteration):
        po_episode_reward = 0
        po_episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > po_total_numsteps:
                po_action = env.action_space.sample()  # Sample random protagonist action
                ad_action = env.action_space.sample()  # Sample random adversary action
            else:
                po_action = po_agent.select_action(state)  # Sample action from po policy
                ad_action = ad_agent.select_action(state)  # Sample action from ad policy

            if len(po_memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = po_agent.update_parameters(po_memory,
                                                                                                         args.batch_size,
                                                                                                         po_updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, po_updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, po_updates)
                    writer.add_scalar('loss/policy', policy_loss, po_updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, po_updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, po_updates)
                    po_updates += 1

            sum_action = po_action + args.ad_factor * ad_action
            sum_action = np.clip(sum_action, env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(sum_action)  # Step
            po_episode_steps += 1
            po_total_numsteps += 1
            po_episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if po_episode_steps == env._max_episode_steps else float(not done)

            po_memory.push(state, po_action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        # if po_total_numsteps > args.num_steps:
        #     break

        writer.add_scalar('reward/train', po_episode_reward, po_episode)
        print("PO Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(po_episode, po_total_numsteps,
                                                                                      po_episode_steps,
                                                                                      round(po_episode_reward, 2)))

        if po_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                po_episode_reward = 0
                done = False
                while not done:
                    po_action = po_agent.select_action(state, evaluate=True)
                    ad_action = ad_agent.select_action(state, evaluate=True)
                    sum_action = po_action + args.ad_factor * ad_action
                    sum_action = np.clip(sum_action, env.action_space.low, env.action_space.high)

                    next_state, reward, done, _ = env.step(sum_action)
                    po_episode_reward += reward

                    state = next_state
                avg_reward += po_episode_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, po_episode)
            print("----------------------------------------")
            print("Test PO Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

        if po_episode % 1000 == 0:  # 1000
            po_agent.save_model(args.env_name, suffix="1021protagonist" + str(args.seed) + "autotune" if args.automatic_entropy_tuning else "")

        po_episode += 1
        total_episode += 1

    for ad_iteration in range(args.ad_iteration):
        ad_episode_reward = 0
        ad_episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > ad_total_numsteps:
                po_action = env.action_space.sample()  # Sample random protagonist action
                ad_action = env.action_space.sample()  # Sample random adversary action
            else:
                po_action = po_agent.select_action(state)  # Sample action from po policy
                ad_action = ad_agent.select_action(state)  # Sample action from ad policy

            if len(ad_memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = ad_agent.update_parameters(ad_memory,
                                                                                                         args.batch_size,
                                                                                                         ad_updates)

                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    ad_updates += 1

            sum_action = po_action + args.ad_factor * ad_action
            sum_action = np.clip(sum_action, env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(sum_action)  # Step
            ad_episode_steps += 1
            ad_total_numsteps += 1
            ad_episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if ad_episode_steps == env._max_episode_steps else float(not done)

            ad_memory.push(state, ad_action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        # if ad_total_numsteps > args.num_steps:
        #     break

        # writer.add_scalar('reward/train', episode_reward, i_episode)
        print("AD Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(ad_episode, ad_total_numsteps,
                                                                                      ad_episode_steps,
                                                                                      round(ad_episode_reward, 2)))

        if ad_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                ad_episode_reward = 0
                done = False
                while not done:
                    po_action = po_agent.select_action(state, evaluate=True)
                    ad_action = ad_agent.select_action(state, evaluate=True)
                    sum_action = po_action + args.ad_factor * ad_action
                    sum_action = np.clip(sum_action, env.action_space.low, env.action_space.high)

                    next_state, reward, done, _ = env.step(sum_action)
                    ad_episode_reward += reward

                    state = next_state
                avg_reward += ad_episode_reward
            avg_reward /= episodes

            # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test AD Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
        if ad_episode % 1000 == 0:  # 1000
            ad_agent.save_model(args.env_name, suffix="1021adversary" + str(args.seed) + "autotune" if args.automatic_entropy_tuning else "")

        ad_episode += 1
        total_episode += 1

    if (po_total_numsteps >= args.num_steps) and (ad_total_numsteps >= args.num_steps):
        break

env.close()

