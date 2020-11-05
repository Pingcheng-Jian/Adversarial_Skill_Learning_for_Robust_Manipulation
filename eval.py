import numpy as np
import gym
from arguments import get_args
from rl_modules.ddpg_agent import ddpg_agent
from rl_modules.sac_agent import sac_agent, sac_br_agent, sac_br_agent_ppo, sac_skill_agent
from rl_modules.double_agent import double_agent
from rl_modules.double_agent_share import double_agent_share
from rl_modules.double_agent_shaping_share import double_agent_shaping_share
from rl_modules.dynamics import dynamics_learner, inverse_dynamics_learner
from rl_modules.double_agent_attention import double_agent_attention
from rl_modules.sgg_agent import sgg_agent
from rl_modules.alternate_agent import alternate_agent
import env_resource.envs.robotics
import random
import torch

PATH = '../env_resource_data/saved_models/DoublePushGoalEnv-v1/actor_lr_0.0001_critic_lr_0.0001/sac_agent_att_success.pt'


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              'attention_head': args.attention_head,
              'hidden_unit': args.hidden_unit,
              'grip':3 if args.only_gripper else 10,
              'n_object': env.n_object,
              'device': args.device,
              }
    params['max_timesteps'] = args.max_episode_steps
    return params


def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)

    if args.algo == 'ddpg':
        eval = ddpg_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'sac':
        eval = sac_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'forward':
        eval = dynamics_learner(args, env, env_params)
        eval.eval_dynamics(PATH)
    elif args.algo == 'inverse':
        eval = inverse_dynamics_learner(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'double':
        eval = double_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'share':
        eval = double_agent_share(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'shaping_share':
        eval = double_agent_shaping_share(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'attention':
        eval = double_agent_attention(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'sgg':
        eval = sgg_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'alt':
        eval = alternate_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'br':
        eval = sac_br_agent(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'br_ppo':
        eval = sac_br_agent_ppo(args, env, env_params)
        eval.visualize(PATH)
    elif args.algo == 'skill':
        eval = sac_skill_agent(args, env, env_params)
        eval.visualize(PATH)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
