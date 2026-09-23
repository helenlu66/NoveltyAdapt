import argparse
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from learning.learning_utils import *
from learning.custom_gym_wrapper import *
from learning.custom_callback import *
from learning.learner import *
from utils import *

if __name__ == '__main__':
    config = load_config("config.yaml")
    parser = argparse.ArgumentParser(description='Train a baseline agent')
    parser.add_argument('--domain', type=str, default='nut_assembly', help='the domain')
    parser.add_argument('--rl_algorithm', type=str, default='PPO', help='the RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='the total number of timesteps to train the agent')
    parser.add_argument('--ep_len', type=int, default=100)
    parser.add_argument('--render_training', type=bool, default=False)
    parser.add_argument('--lr_schedule', action='store_true')
    parser.set_defaults(lr_schedule=False)  # Default value
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replay_buffer', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=False)  # Default value
    args = parser.parse_args()

    # Set the random seed for reproducibility
    config['learning'][args.rl_algorithm]['seed'] = args.seed

    # if there's learning rate schedule, replace the learning rate in the config file
    if args.lr_schedule:
        original_lr = config['learning'][args.rl_algorithm]['learning_rate']
        config['learning'][args.rl_algorithm]['learning_rate'] = linear_schedule(original_lr)

    env = load_env(domain=args.domain, config=config['simulation'])

    learner = OperatorDiscoverer(env=env, args=args, config=config)
    if args.replay_buffer:
        learner.save_path = learner.save_path + "_replay_buffer"
    learner.discover_operator()