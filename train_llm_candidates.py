import os
import gymnasium as gym
import importlib
import numpy as np
import logging
import argparse
import json
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import SAC, PPO, DDPG
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
    parser.add_argument('--domain', choices=['nut_assembly', 'kitchen', 'coffee_drawer', 'coffee_box'], default='nut_assembly', help='the domain')
    parser.add_argument('--rl_algorithm', type=str, default='PPO', help='the RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=config['learning']['learn_subgoal']['total_timesteps'])
    parser.add_argument('--timesteps_per_iter', type=int, default=config['learning']['learn_subgoal']['timesteps_per_iter'])
    parser.add_argument('--ep_len', type=int, default=100)
    parser.add_argument('--render_training', type=bool, default=False)
    parser.add_argument('--lr_schedule', action='store_true')
    parser.set_defaults(lr_schedule=False)  # Default value
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replay_buffer', action='store_true')
    parser.add_argument('--rw_type', choices=['pos_linear_dist', 'pos_linear_vel', 'pos_exp_dist', 'pos_exp_dist_test', 'pos_exp_vel', 'pos_linear_vel_test', 'pos_linear_vel_test_range', 'neg_exp_dist', 'neg_exp_vel'], default='pos_linear_vel', help='the type of reward shaping function to use')
    # parser.add_argument('--terminate_on_success', action='store_true', help='whether to terminate the episode on success. If false, the subgoal bonus will be smaller as powers of 5.')
    parser.add_argument('--termination_type', choices=['on_first_subgoal_success', 'on_all_subgoal_success', 'no_termination', 'on_non_terminal_subgoal_success'], default='on_first_subgoal_success', help='whether to terminate the episode on success. If false, the subgoal bonus will be smaller as powers of 10.')
    #parser.set_defaults(terminate_on_success=False)  # Default value
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=False)  # Default value
    args = parser.parse_args()

    # Set the random seed for reproducibility
    config['learning'][args.rl_algorithm]['seed'] = args.seed

    # if there's learning rate schedule, replace the learning rate in the config file
    if args.lr_schedule:
        original_lr = config['learning'][args.rl_algorithm]['learning_rate']
        config['learning'][args.rl_algorithm]['learning_rate'] = linear_schedule(original_lr)

    grounded_op = choose_grounded_op(config, args.domain)
    env = load_env(domain=args.domain, config=config['simulation'])
    learner = LLMSubgoalCurriculumLearner(env=env, args=args, grounded_operator_to_learn=grounded_op, config=config)
    if args.replay_buffer:
        learner.save_path = learner.save_path + "_replay_buffer"
    learner.learn_operator()