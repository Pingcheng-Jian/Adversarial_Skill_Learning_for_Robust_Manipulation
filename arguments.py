import argparse

"""
Here are the param for the training
"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--logdir', type=str, default='logdir/', help='the tensorboard file name')
    parser.add_argument('--algo', type=str, default='sac', help='the algorithm name')
    parser.add_argument('--env-name', type=str, default='DoublePushEnv-v3', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=300, help='the number of epochs to train the agent')  # 50
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')  # originally 50
    parser.add_argument('--n-batches', type=int, default=50, help='the times to update the network')
    parser.add_argument('--n-dynamics', type=int, default=50, help='the times to update the dynamics')
    parser.add_argument('--n-batches-sgg', type=int, default=20,
                        help='the times to update the network using subgoal samples')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=24, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')  # future
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--save-name', type=str, default='sac', help='the path to save the models')
    parser.add_argument('--save-training-success-rate-dir', type=str, default='saved_training_success_rate/', help='the path to save the training success rate')
    parser.add_argument('--save-training-return-dir', type=str, default='saved_training_return/',
                        help='the path to save the training return')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')  # 80% replace
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=1e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=1e-4, help='the learning rate of the critic')
    # for adversarial training
    parser.add_argument('--lr-ad_actor', type=float, default=1e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-ad_critic', type=float, default=1e-4, help='the learning rate of the critic')
    parser.add_argument('--ad_factor', type=float, default=0.2, help='the amplitude of adversary agent')
    parser.add_argument('--test_ad_factor', type=float, default=0.2, help='the amplitude of adversary agent')
    parser.add_argument('--outer-iteration', type=int, default=20,
                        help='the number of outer iteration for adversarial training')
    parser.add_argument('--po-iteration', type=int, default=5,
                        help='the number of inner iteration for adversarial training of protagonist')
    parser.add_argument('--ad-iteration', type=int, default=5,
                        help='the number of inner iteration for adversarial training of adversary')
    parser.add_argument('--ad-alpha', type=float, default=0.2, help='alpha of adversarial sac')
    parser.add_argument('--ad_alpha_lr', type=float, default=3e-4, help='adversarial alpha lr of sac')
    # parser.add_argument('--test-iteration', type=int, default=50, help='the number of test iteration for testing')
    # #
    parser.add_argument('--lr-actor_2', type=float, default=1e-4, help='the learning rate of the actor 2')
    parser.add_argument('--lr-critic_2', type=float, default=1e-4, help='the learning rate of the critic 2')
    parser.add_argument('--lr_dyna', type=float, default=1e-3, help='the learning rate of dynamics')
    parser.add_argument('--test-iteration', type=int, default=50, help='the number of test iteration for testing')
    # above is added for adversarial training
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', type=bool, default=False, help='if use gpu do the acceleration')  # True
    parser.add_argument('--device', type=int, default=0, help='which gpu to use')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha of sac')
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='alpha lr of sac')
    parser.add_argument('--count_exp', type=bool, default=False, help='if use count-based exploration')
    parser.add_argument('--exp_beta', type=float, default=0.2, help='weight of count-based exploration')
    parser.add_argument('--share_reward', type=bool, default=False, help='if use shared reward for double agent')
    parser.add_argument('--tensorboard', type=bool, default=False, help='if use tb')
    parser.add_argument('--max_episode_steps', type=int, default=50, help='time steps')
    parser.add_argument('--attention_head', type=int, default=1, help='number of heads in attention module')
    parser.add_argument('--hidden_unit', type=int, default=512, help='number of hidden units')
    parser.add_argument('--only_for_door', type=bool, default=True,
                        help='let the reward of another arm be in only-for-door version')
    parser.add_argument('--lr-value', type=float, default=1e-4, help='value network lr')
    parser.add_argument('--subgoal_k', type=int, default=3, help='number of subgoal samples')
    parser.add_argument('--subgoal_start', type=int, default=50, help='subgoal sampling start epoch')
    parser.add_argument('--random_object', type=bool, default=False,
                        help='if randomize object position when sampling subgoal')
    parser.add_argument('--must_success', type=bool, default=False)
    parser.add_argument('--behavior_cloning', type=bool, default=False)
    parser.add_argument('--sil', type=bool, default=False, help='if use self imitation learning')
    parser.add_argument('--sil_beta', type=float, default=1.0, help='self imitation learning rate')
    parser.add_argument('--stop_p', type=float, default=0.1, help='action stop probability')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='clip for ppo')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='clip grad norm for ppo')
    parser.add_argument('--stop', type=bool, default=False, help='if execute stopping')
    parser.add_argument('--ppo_alpha', type=float, default=0.02, help='alpha for ppo dist')
    parser.add_argument('--option_step', type=int, default=5, help='option step length')
    parser.add_argument('--lr-skill', type=float, default=3e-4, help='lr for skill selection')
    parser.add_argument('--detector', type=str, default='distance', help='how to detect key state')
    parser.add_argument('--key-dist', type=float, default=0.1, help='lower bound of key dist')
    parser.add_argument('--sac-attention', type=bool, default=False, help='if use attention-based sac')
    parser.add_argument('--only-gripper', type=bool, default=False, help='if only predict gripper pos')
    parser.add_argument('--bc_w', type=float, default=1., help='weight for behavior cloning')

    args = parser.parse_args()
    return args


def get_dyna_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--logdir', type=str, default='logdir/', help='the tensorboard file name')
    parser.add_argument('--algo', type=str, default='forward', help='the algorithm name')
    parser.add_argument('--env-name', type=str, default='PushEnv-v0', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=200, help='the number of epochs to train the agent')  # 50
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')  # originally 50
    parser.add_argument('--n-batches', type=int, default=500, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=24, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')  # future
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e5), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')  # 80% replace
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=64, help='the sample batch size')
    parser.add_argument('--test-iteration', type=int, default=50, help='the number of test iteration for testing')
    parser.add_argument('--lr_rew', type=float, default=1e-3, help='the learning rate of reward')
    parser.add_argument('--lr_dyna', type=float, default=1e-3, help='the learning rate of dynamics')
    parser.add_argument('--lr_inverse_dyna', type=float, default=5e-5, help='the learning rate of inverse dynamics')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', type=bool, default=True, help='if use gpu do the acceleration')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--tensorboard', type=bool, default=True, help='if use tb')

    args = parser.parse_args()

    return args


def get_eval_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--algo', type=str, default='adversarial_double_stack_sac', help='the algorithm name')
    parser.add_argument('--env-name', type=str, default='MultiFetchStackEasyEnv-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=200, help='the number of epochs to train the agent')  # 50
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')  # originally 50
    parser.add_argument('--n-batches', type=int, default=500, help='the times to update the network')
    parser.add_argument('--n-batches-sgg', type=int, default=20,
                        help='the times to update the network using subgoal samples')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=24, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')  # future
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e5), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')  # 80% replace
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=64, help='the sample batch size')
    parser.add_argument('--test-iteration', type=int, default=50, help='the number of test iteration for testing')
    parser.add_argument('--lr-actor', type=float, default=1e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=1e-4, help='the learning rate of the critic')
    parser.add_argument('--lr_rew', type=float, default=1e-3, help='the learning rate of reward')
    parser.add_argument('--lr_dyna', type=float, default=1e-3, help='the learning rate of dynamics')
    parser.add_argument('--lr_inverse_dyna', type=float, default=5e-5, help='the learning rate of inverse dynamics')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', type=bool, default=False, help='if use gpu do the acceleration')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha of sac')
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='alpha lr of sac')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--tensorboard', type=bool, default=False, help='if use tb')
    parser.add_argument('--max_episode_steps', type=int, default=100, help='time steps')
    parser.add_argument('--attention_head', type=int, default=1, help='number of heads in attention module')
    parser.add_argument('--hidden_unit', type=int, default=512, help='number of hidden units')
    parser.add_argument('--only_for_door', type=bool, default=True,
                        help='let the reward of another arm be in only-for-door version')
    parser.add_argument('--lr-value', type=float, default=1e-4, help='value network lr')
    parser.add_argument('--subgoal_k', type=int, default=3, help='')
    parser.add_argument('--subgoal_start', type=int, default=50, help='subgoal sampling start epoch')
    parser.add_argument('--random_object', type=bool, default=False,
                        help='if randomize object position when sampling subgoal')
    parser.add_argument('--sil', type=bool, default=False, help='if use self imitation learning')
    parser.add_argument('--sil_beta', type=float, default=1.0, help='self imitation learning rate')
    parser.add_argument('--must_success', type=bool, default=True, help='if only collect successful trajectory')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='clip for ppo')
    parser.add_argument('--stop_p', type=float, default=0.1, help='action stop probability')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='clip grad norm for ppo')
    parser.add_argument('--stop', type=bool, default=False, help='if execute stopping')
    parser.add_argument('--ppo_alpha', type=float, default=0.02, help='alpha for ppo dist')
    parser.add_argument('--option_step', type=int, default=5, help='option step length')
    parser.add_argument('--lr-skill', type=float, default=3e-4, help='lr for skill selection')
    parser.add_argument('--detector', type=str, default='distance', help='how to detect key state')
    parser.add_argument('--key-dist', type=float, default=0.1, help='lower bound of key dist')
    parser.add_argument('--sac-attention', type=bool, default=False, help='if use attention-based sac')
    parser.add_argument('--only-gripper', type=bool, default=False, help='if only predict gripper pos')
    parser.add_argument('--bc_w', type=float, default=1., help='weight for behavior cloning')

    args = parser.parse_args()

    return args
