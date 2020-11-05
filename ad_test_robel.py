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
parser.add_argument('--ad_factor', type=float, default=0.0,
                    help='adversary action amplitude')
parser.add_argument('--po_agent', default="robust",
                    help='robust or normal')

args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
# env = gym.make(args.env_name, device_path='/dev/ttyUSB0')

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
po_agent = SAC(env.observation_space.shape[0], env.action_space, args)
ad_agent = Adv_SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
# writer = SummaryWriter('runs/Adv_Train_{}_SAC_{}_{}_{}_{}_ad{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#                                                      args.env_name,
#                                                      args.policy,
#                                                      "autotune" if args.automatic_entropy_tuning else "",
#                                                      args.seed,
#                                                      args.ad_factor))

# Memory
po_memory = ReplayMemory(args.replay_size, args.seed)
ad_memory = ReplayMemory(args.replay_size, args.seed)

normal_po_agent_actor_path = '/home/robot/github/tmp/models/sac_actor_DClawTurnFixed-v0_456autotune'
normal_po_agent_critic_path = '/home/robot/github/tmp/models/sac_critic_DClawTurnFixed-v0_456autotune'

if args.po_agent == 'robust':
    po_agent.load_model('models/po_sac_actor_DClawTurnFixed-v0_1021protagonist121autotune', 'models/po_sac_critic_DClawTurnFixed-v0_1021protagonist121autotune')

    # po_agent.load_model('saved_models/robel/normal-sac/Normal_sac_actor_DClawTurnFixed-v0_-1-123456autotune',
    #                     'saved_models/robel/normal-sac/Normal_sac_critic_DClawTurnFixed-v0_-1-123456autotune')
else:
    po_agent.load_model(normal_po_agent_actor_path, normal_po_agent_critic_path)

ad_agent.load_model('models/ad_sac_actor_DClawTurnFixed-v0_1021adversary121autotune', 'models/ad_sac_critic_DClawTurnFixed-v0_1021adversary121autotune')

reward_list = []

for ite in range(1):
    args.seed = 121
    episodes = 1
    for episode in range(episodes):
        po_action_list = []
        ad_action_list = []
        angle_list = []
        avg_reward = 0.
        state = env.reset()
        angle_list.append(state[-1])
        ad_episode_reward = 0
        done = False
        while not done:
            po_action = po_agent.select_action(state, evaluate=True)
            ad_action = ad_agent.select_action(state, evaluate=True)
            sum_action = po_action + args.ad_factor * ad_action
            po_action_list.append(po_action[6])
            ad_action_list.append(ad_action[6])
            sum_action = np.clip(sum_action, env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(sum_action)
            angle_list.append(next_state[-1])
            # env.render
            ad_episode_reward += reward

            state = next_state
        avg_reward += ad_episode_reward
        # print(angle_list)
        # np.save('saved_robel_angle/'+args.po_agent+'_adv_ad'+str(args.ad_factor)+'_'+str(episode)+'.npy', np.array(angle_list))
        avg_reward /= episodes
        print(po_action_list)
        print(ad_action_list)
        print(avg_reward)
        np.save('saved_robel_action/oursVSadv_po_action_ad08_6', po_action_list)
        np.save('saved_robel_action/oursVSadv_ad_action_ad08_6', ad_action_list)

    # reward_list.append(avg_reward)
    # print(avg_reward)

# reward_list = np.array(reward_list)
# mean_reward = np.mean(np.array(reward_list))
# reward_std = np.std(np.array(reward_list))
# print(reward_list)
# np.save('saved_robel_reward/normal_vs_AdvAttack_0_005_135', reward_list)
