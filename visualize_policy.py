# A minimal script to visualize the grasp policy trained with PPO in the `debug_training.py` script

import copy
import time
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import argparse
from gymnasium.wrappers.time_limit import TimeLimit
from learning.custom_gym_wrapper import *
from tarski import fstrips as fs
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from utils import *

def choose_reward_shaping_fn(op_wrap:OperatorWrapper) -> OperatorWrapper:
    """Prints out the reward shaping functions and returns the selected reward shaping function

    Args:
        op_wrap (OperatorWrapper): The OperatorWrapper whose subgoal reward shaping functions need to be se

    Returns:
        the same OperatorWrapper with the selected reward shaping functions set
    """
    # select the reward shaping file for each effect in the grounded operator
    print("Here is a list of reward shaping functions for each effect in the grounded operator:\n")
    op_name, _ = extract_name_params_from_grounded(op_wrap.grounded_operator.ident())
    for effect in op_wrap.grounded_operator.effects:
        print(f"Reward shaping functions for {effect.pddl_repr()}:")
        for i, file_name in enumerate(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")):
            if "__pycache__" in file_name:
                continue
            if op_name in file_name:
                print(f"[{i}] {file_name}")
        try:
            s = input("Choose a reward shaping function for the effect " + "(enter a number from 0 to {}): ".format(len(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")) - 1))
            # parse input into a number within range
            k = min(max(int(s), 0), len(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")))
        except:
            k = 0
            print("Input is not valid. Use {} by default.\n".format(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")[k]))
        # drop the .py extension
        reward_shaping_fn_file_name = os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")[k].split(".")[0]
    
        reward_shaping_fn_module = importlib.import_module(f"learning.reward_functions.{op_wrap.args.domain}.{reward_shaping_fn_file_name}")
        reward_shaping_func = getattr(reward_shaping_fn_module, 'reward_shaping_fn')
        op_wrap.set_subgoal_reward_shaping_fn(effect, reward_shaping_func)
        print()
    return op_wrap

if __name__ == '__main__':
    # parse model commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_wrap', action='store_true', help='Enable operator wrapper')
    parser.set_defaults(op_wrap=False)  # Default value
    parser.add_argument('--llm', action='store_true', help='Enable LLM reward shaping')
    parser.set_defaults(llm=False)  # Default value
    parser.add_argument('--domain', type=str, default='coffee', help='the domain of the environment')
    parser.add_argument('--grounded_op', type=str, default='open_drawer1', help='the name of the operator')
    parser.add_argument('--seed', type=int, default=0, help='the seed for the environment')
    parser.add_argument('--rw_shaping', type=int, default=0, help='the reward shaping function to use')
    parser.add_argument('--rl_algorithm', type=str, default='PPO', help='the algorithm to use')
    parser.add_argument('--ep_len', type=int, default=100, help='the length of the episode')
    parser.add_argument('--replay_buffer', action='store_true', help='Enable replay buffer')
    parser.add_argument('--rw_type', choices=['pos_linear_dist', 'pos_linear_vel', 'pos_exp_dist', 'pos_exp_vel','neg_exp_dist', 'neg_exp_vel'], default='pos_exp_dist', help='the type of reward shaping function to use')
    parser.set_defaults(replay_buffer=False)  # Default value
    parser.add_argument('--termination_type', choices=['on_first_subgoal_success', 'on_all_subgoal_success', 'no_termination', 'on_non_terminal_subgoal_success'], default='on_first_subgoal_success', help='whether to terminate the episode on success. If false, the subgoal bonus will be smaller as powers of 10.')
    parser.set_defaults(terminate_on_success=False)  # Default value
    args = parser.parse_args()

    np.random.seed(args.seed)
    config:dict = load_config("config.yaml")
    domain:str = args.domain
    config['simulation']['has_renderer'] = True
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    robosuite_env.viewer.set_camera(camera_id=3)
    plan = load_plan(config['planning'][domain]) # load the plan in case the operator wrapper is used
    grounded_op = choose_grounded_op(config, domain)
    # grounded_op = find_grounded_operator_from_plan(plan, args.grounded_op)
    op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
    curr_subgoal = choose_subgoal(grounded_op)
    filtered_obs = filter_observation(get_observation_with_semantics(robosuite_env), grounded_op)
    gym_env = GymWrapper(visual_env, keys=list(filtered_obs.keys()))
    gym_env = TimeLimit(gym_env, max_episode_steps=args.ep_len)
    if args.op_wrap:
        # load LLM generated reward function in case the operator wrapper is used
        reward_fn_candidates = []
        if args.llm:
            for i in range(config['learning']['reward_shaping_fn']['num_candidates']):
                try:
                    llm_reward_shaping_func = load_llm_reward_shaping_fn_candidate(domain=domain, op_name=op_name, candidate_idx=i, rw_type=args.rw_type)
                    reward_fn_candidates.append(llm_reward_shaping_func)
                except Exception as e:
                    raise Exception(e)
            wrapped_env = CollisionAblatedOperatorWrapper(gym_env, grounded_operator=grounded_op, config=config, args=args, curr_subgoal=curr_subgoal)
            choose_reward_shaping_fn(wrapped_env)
            #wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
        else:
            wrapped_env = CollisionLLMAblatedOperatorWrapper(gym_env, config=config, grounded_operator=grounded_op, args=args, curr_subgoal=curr_subgoal)
    else:
        wrapped_env = MinimalWrapper(gym_env, config, domain)
    env = Monitor(wrapped_env, filename=f'{args.domain}_visualize_monitor', allow_early_resets=True)

    rl_algo = importlib.import_module(f"stable_baselines3.{args.rl_algorithm.lower()}").__dict__[args.rl_algorithm.upper()]
    op_name = op_name.replace('(', '_').replace(')', '_').replace(' ', '_')
    subgoal_name = curr_subgoal.pddl_repr().replace(' ', '_')
    if args.llm:
        if args.replay_buffer:
            model = rl_algo.load(f'./learning/policies/{args.domain}/llm_subgoal/{op_name}/{args.rl_algorithm}/seed_{args.seed}_replay_buffer/{subgoal_name}/reward_fn_{args.rw_shaping}/best_model/best_model', env=env)
        else:
            model = rl_algo.load(f'./learning/policies/{args.domain}/llm_subgoal/{op_name}/{args.rl_algorithm}/seed_{args.seed}/{subgoal_name}/reward_fn_{args.rw_shaping}/best_model/best_model', env=env)
    else:
        if args.replay_buffer:
            model = rl_algo.load(f'./learning/policies/{args.domain}/subgoal/{op_name}/{args.rl_algorithm}/seed_{args.seed}_replay_buffer/{subgoal_name}/best_model/best_model', env=env)
        else:
            model = rl_algo.load(f'./learning/policies/{args.domain}/subgoal/{op_name}/{args.rl_algorithm}/seed_{args.seed}/{subgoal_name}/best_model/best_model', env=env)

    obs, info = env.reset()
    detector = load_detector(config, args.domain, env)
    for _ in range(config['learning']['eval']['n_eval_episodes']):
        done, truncated = False, False
        steps = 0
        camera_id = 2
        print(robosuite_env.sim.model.camera_names)
        print(robosuite_env.sim.model.camera_names[camera_id])
        robosuite_env.viewer.set_camera(camera_id=camera_id)
        visual_env.render()
        while not (done or truncated):
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            binary_obs = detector.detect_binary_states(env)
            obs_with_semantics = get_observation_with_semantics(env.unwrapped)
            subgoal_achieved = check_condition_satisfied(curr_subgoal, binary_obs)
            for i, effect in enumerate(wrapped_env.grounded_operator.effects):
                if effect.atom.pddl_repr() in binary_obs:
                    print(effect.atom.pddl_repr(), ":", binary_obs[effect.atom.pddl_repr()])
            print("action: ", action)
            print('reward: ', reward)
            print(obs)
            steps += 1
            visual_env.render()
        obs, info = env.reset()
    env.close()