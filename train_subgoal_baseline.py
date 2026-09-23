
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
    parser.add_argument('--domain', choices=['nut_assembly', 'kitchen', 'coffee_drawer', 'coffee_box'], default='nut_assembly', help='the domain')
    parser.add_argument('--rl_algorithm', type=str, default='PPO', help='the RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=config['learning']['learn_subgoal']['total_timesteps'])
    parser.add_argument('--rw_type', choices=['step_cost', 'pos_subgoal_sparse'], default='pos_subgoal_sparse', help='the type of reward shaping function to use')
    parser.add_argument('--termination_type', choices=['on_first_subgoal_success', 'on_all_subgoal_success', 'no_termination', 'on_non_terminal_subgoal_success'], default='on_first_subgoal_success', help='whether to terminate the episode on success. If false, the subgoal bonus will be smaller as powers of 10.')
    parser.add_argument('--ep_len', type=int, default=100)
    parser.add_argument('--render_training', type=bool, default=False)
    parser.add_argument('--lr_schedule', action='store_true')
    parser.set_defaults(lr_schedule=False)  # Default value
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replay_buffer', action='store_true')
    parser.set_defaults(replay_buffer=False)  # Default value
    args = parser.parse_args()

    # Set the random seed for reproducibility
    config['learning'][args.rl_algorithm]['seed'] = args.seed

    # if there's learning rate schedule, replace the learning rate in the config file
    if args.lr_schedule:
        original_lr = config['learning'][args.rl_algorithm]['learning_rate']
        config['learning'][args.rl_algorithm]['learning_rate'] = linear_schedule(original_lr)

    grounded_op = choose_grounded_op(config, args.domain)
    env = load_env(domain=args.domain, config=config['simulation'])
    # scale the total timesteps based on the number of llm candidates' elimination criteria so that the operator baseline gets the same amount of training as all the llm candidates combined
    args.total_timesteps = args.total_timesteps + config['learning']['learn_subgoal']['timesteps_per_iter'] + 2 * config['learning']['learn_subgoal']['timesteps_per_iter'] # one candidate will be eliminated in the first iteration of training, and another in the second iteration. One candidate will remain to be trained for the total timesteps
    env = load_env(domain=args.domain, config=config['simulation'])
    learner = SubgoalCurriculumLearner(env=env, args=args, grounded_operator_to_learn=grounded_op, config=config)
    if args.replay_buffer:
        learner.save_path = learner.save_path + "_replay_buffer"
    learner.learn_operator()