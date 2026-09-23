
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import csv
import argparse
from tarski import fstrips as fs
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from learning.learning_utils import *
from utils import *
from planning.hybrid_symbolic_llm_planner import HybridSymbolicLLMPlanner
from planning.planning_utils import make_root_node
from tarski.search import GroundForwardSearchModel
from tarski.model import Model
from tarski.syntax.predicate import Predicate
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators


class OperatorDiscoveryWrapper(gym.Wrapper):
    def __init__(self, env:MujocoEnv, config:dict, args:argparse.Namespace, save_path:str=None):
        super().__init__(env)
        self.config = config
        self.args = args
        self.detector = load_detector(config=config, domain=self.args.domain, env=env)
        self.episode_r_shaping = 0
        self.time_step = 0
        self.episode = -1
        self.initial_symbols = None # the initial symbols in the environment, compared with the final set of symbols to determine the discovered operator
        self.change_trace = [] # list of dictionaries that contain the changed symbols trace
        self.discovered_operator = None # the operator discovered in the episode
        if save_path is not None:
            log_dir = os.path.join(project_root, save_path)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'OperatorDiscoveryChecker.log')
        self.planner = HybridSymbolicLLMPlanner(config=self.config['planning'][self.args.domain], logger_path=log_path)
        self.start_model= GroundForwardSearchModel(self.planner.starting_problem, ground_problem_schemas_into_plain_operators(self.planner.starting_problem))
        self.root = copy.deepcopy(self.start_model.init())
        self.time_step = 0

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """step function that steps the environment and computes the reward based on the observation with semantics

        Args:
            action (array): 7 dimensional array representing the action to take

        Returns:
            Tuple[np.array, float, bool, bool, dict]: the observation, reward, done, truncated, and info
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        action = self._discretize_gripper_action(action)
        # get the observation with semantics before taking the action, make sure it does not get modified by the step function
        prev_obs_with_semantics:dict = self.detector.get_obs()
        # preserve the values of the dictionary so they are not modified by the step function
        prev_obs_with_semantics = {k: v.copy() for k, v in prev_obs_with_semantics.items()}
        prev_binary_obs = self.detector.detect_binary_states(self.env.unwrapped)
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        curr_binary_obs:dict = self.detector.detect_binary_states(self.env.unwrapped)
        reward = self.compute_reward(prev_binary_obs=prev_binary_obs, curr_binary_obs=curr_binary_obs)
        
        # save reward and penalties separately
        self.episode_r_shaping += reward

        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['num_symbol_changes'] = len(self.change_trace[-1]) if len(self.change_trace) > 0 else 0 # the number of symbol changes in the last step
        info['discovered_operator'] = self.discovered_operator # the operator discovered in the episode
        info['changed_symbols'] = self.change_trace # the symbols that have changed since the beginning of the episode
    
        done = self.discovered_operator is not None # the episode is done if an operator is discovered
        
        self.time_step += 1
        return obs, reward, done, truncated, info    

    def reset(self, **kwargs):
        
        # first, reset the environment to the very beginning
        try: # kwargs include the seed
            obs, info = self.env.reset(**kwargs)
            # step the environment a few steps so the lid falls on the pot
            for _ in range(10):
                obs, _, _, _, _ = self.env.step(np.zeros(self.env.action_space.shape))
        except:
            obs = self.env.reset(**kwargs)
            info = {}
        # set the timestep to 0

        self.unwrapped.timestep=0 # reset the timesteps
        self.time_step = 0
        if hasattr(self.env, '_elapsed_steps'):
            self.env._elapsed_steps = 0 # reset the elapsed steps
            # reset the episode rewards and penalties
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        self.change_trace = [] # reset the change trace
        self.initial_symbols = None # reset the initial symbols
        self.discovered_operator = None # reset the discovered operator
        self.root = copy.deepcopy(self.start_model.init()) # reset the root node of the search model
        
        self.detector.update_obs()
        self.episode += 1
        # reset the infos
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['num_symbol_changes'] = 0 # reset the number of symbol changes
        info['discovered_operator'] = None # reset the discovered operator flag
        info['changed_symbols'] = [] # reset the changed symbols
        
        
        info['episode'] = { # for compatibility with the stable_baselines3's update_info_buffer
            'ep_cumu_r_shaping': self.episode_r_shaping,
            'num_symbol_changes': 0, # reset the number of symbol changes
            'discovered_operator': None, # reset the discovered operator flag
            'changed_symbols': [] # reset the changed symbols
        }
        self.time_step = 0
        return obs, info
        
    def compute_reward(self, prev_binary_obs:dict, curr_binary_obs:dict) -> float:
        """give a discovery reward for finding an operator

        Args:
            prev_binary_obs (dict): the observation before taking the action in which the keys are predicates and the values are True/False
            curr_binary_obs (dict): the current observation after taking the action in which the keys are predicates and the values are True/False

        Returns:
            float: the reward
        """
        # if this is the first step, cache the initial symbols
        if self.initial_symbols is None:
            self.initial_symbols = curr_binary_obs.copy()
            return 0
        
        # find the changed symbols in the current episode
        diff_symbols_from_initial = {}
        changed_from_prev = {}
        for key, value in curr_binary_obs.items():
            if key not in prev_binary_obs or prev_binary_obs[key] != value:
                changed_from_prev[key] = value
            if self.initial_symbols is not None and self.initial_symbols[key] != value:
                diff_symbols_from_initial[key] = value
        
        # if there are changees from prev, append it to trace
        if len(changed_from_prev) > 0:
            self.change_trace.append(changed_from_prev)
        # if it's the same as the initial state, return 0 reward
        if len(diff_symbols_from_initial) == 0:
            return 0
        
        # if there are no changed symbols, return 0 reward
        if len(changed_from_prev) == 0:
            return 0
        
        # otherise, use launch the symbolic planner to see if it can find a path to goal from the current state
        
        #root = self.start_model.init()
        language = self.root.language
        for key, value in changed_from_prev.items():
            # split at space, the first is the predicate name, the rest are the parameters
            parts = key.split(' ')
            pred_name = parts[0]
            params = parts[1:] if len(parts) > 1 else []
            params = [language.get_constant(p) for p in params]  # convert to Sort objects
            # create a Predicate object
            pred = language.get_predicate(pred_name)
            if value: # if the symbol is True, add it to the root
                # add the predicate to the root
                self.root.add(pred, *params)
            else: # if the symbol is False, remove it from the root
                # if the params are not in the predicate, skip it
                try:
                    self.root.remove(pred, *params)
                except KeyError:
                    pass
        #self.start_model.problem.init = root
        plan = self.planner.search_ahead(problem=self.start_model.problem, start_node=make_root_node(self.root), max_depth=self.planner.max_depth)
        # check if plan exists
        if plan is None:
            return 0
        # if plan exists, we have discovered an operator
        self.discovered_operator = diff_symbols_from_initial
        return 10
    
    def _discretize_gripper_action(self, action:np.array) -> np.array:
        """discretize the gripper opening action into 3 discrete actions: open, close, and do nothing

        Args:
            action (np.array): the action to discretize

        Returns:
            np.array: the discretized action
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        gripper_opening_min = self.env.action_space.low[-1]
        gripper_opening_max = self.env.action_space.high[-1]
        gripper_opening_range = gripper_opening_max - gripper_opening_min
        gripper_close_threshold = gripper_opening_min + gripper_opening_range/3
        gripper_open_threshold = gripper_opening_max - gripper_opening_range/3
        if action[-1] < gripper_close_threshold: # close the gripper
            action[-1] = gripper_opening_min
        elif action[-1] > gripper_open_threshold: # open the gripper
            action[-1] = gripper_opening_max
        else: # do nothing
            action[-1] = 0
        return action
        
        
        

class OperatorWrapper(gym.Wrapper):
    def __init__(self, env:MujocoEnv, config:dict, args:argparse.Namespace, grounded_operator:fs.Action, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env)
        self.grounded_operator = grounded_operator
        self.config = config
        self.args = args
        self.curr_subgoal = curr_subgoal
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = find_prev_operators_executors(self.config, self.args.domain, self.grounded_operator)
        self.detector = load_detector(config=config, domain=self.args.domain, env=env)
        self.subgoal_successes, self.episode_historical_subgoal_successes, self.subgoal_reward_shaping_fn_mapping = self._init_subgoal_dicts()
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        self.time_step = 0
        self.episode = -1
        self.record_rollouts = record_rollouts
        self.reward_range = (0, float('inf'))
        self.num_subgoal_in_focus_success_steps = 0 # number of steps the current subgoal in focus has been successfully achieved

        if self.record_rollouts:
            self.rollout_save_dir = f"learning/policies/{self.args.domain}/{op_name}/seed_{self.config['learning'][self.rl_algo_name]['seed']}"
            _, largest_file_number = find_file_with_largest_number(self.rollout_save_dir, 'rollout')
            if largest_file_number is None:
                largest_file_number = 0
            self.rollout_save_path = f"{self.rollout_save_dir}/rollout_{largest_file_number+1}.csv"
            self.csv_file = open(self.rollout_save_path, 'w')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['gripper_x', 'gripper_y', 'gripper_z', 'closest_x', 'closest_y', 'closest_z', 'collision_penalty', 'total_reward', 'num_achieved_subgoals', 'done', 'timestep', 'episode'])
        
    def _init_subgoal_dicts(self) -> Tuple[OrderedDict[str, bool], OrderedDict[str, bool], OrderedDict[str, Callable]]:
        """initialize the subgoal dictionaries for recording the subgoal successes of the last step and the mapping between subgoals and their reward shaping functions
        Returns:
            Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Callable]]: the subgoal success dictionary and the subgoal reward shaping function mapping
        """
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        duplicate_grasp_effects = check_duplicate_grasp_effects(self.grounded_operator)
        duplicate_ungrasp_effects = check_duplicate_free_gripper_effects(self.grounded_operator)
        subgoal_success_dict = OrderedDict()
        episode_historical_subgoal_successes = OrderedDict() # whether the subgoal was achieved at least once in the episode
        subgoal_reward_shaping_fn_mapping = OrderedDict()
        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            if effect.pddl_repr() == 'free gripper1' and duplicate_ungrasp_effects: # if the effect is `free gripper1`, skip it since it is the same effect as `not (exclusively-occupying-gripper ?object gripper1)`
                continue
            subgoal_success_dict[effect.pddl_repr()] = False
            episode_historical_subgoal_successes[effect.pddl_repr()] = False
            subgoal_reward_shaping_fn_mapping[effect.pddl_repr()] = None
        return subgoal_success_dict, episode_historical_subgoal_successes, subgoal_reward_shaping_fn_mapping
    
    def set_subgoal_reward_shaping_fn(self, effect:Union[str, fs.SingleEffect], fn:Callable):
        """Set the reward shaping function for the subgoal

        Args:
            effect (str): the subgoal
            fn (Callable): the reward shaping function
        """
        if isinstance(effect, fs.SingleEffect):
            effect = effect.pddl_repr()
        self.subgoal_reward_shaping_fn_mapping[effect] = fn


    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """step function that steps the environment and computes the reward based on the observation with semantics

        Args:
            action (array): 7 dimensional array representing the action to take

        Returns:
            Tuple[np.array, float, bool, bool, dict]: the observation, reward, done, truncated, and info
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        action = self._discretize_gripper_action(action)
        # get the observation with semantics before taking the action, make sure it does not get modified by the step function
        prev_obs_with_semantics:dict = self.detector.get_obs()
        # preserve the values of the dictionary so they are not modified by the step function
        prev_obs_with_semantics = {k: v.copy() for k, v in prev_obs_with_semantics.items()}
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env.unwrapped)
        reward = self.compute_reward(prev_numeric_obs_with_semantics=prev_obs_with_semantics, current_numeric_obs_with_semantics=obs_with_semantics, binary_obs=binary_obs)
        penalties, collision_points = self.collision_penalty(obs_with_semantics)
        # save reward and penalties separately
        self.episode_r_shaping += reward
        self.episode_collision_penalty += sum(penalties)
        self.episode_num_collisions += len(collision_points)
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = self.subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        subgoal_successes = []
        for subgoal, success in self.subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
            subgoal_successes.append(success)
        info['goal_progress'] = subgoal_successes

        historical_subgoal_successes = []
        for subgoal, historical_success in self.episode_historical_subgoal_successes.items():
            info[f'{subgoal}_historical_subgoal'] = historical_success
            historical_subgoal_successes.append(historical_success)
        info['historical_subgoal_successes'] = historical_subgoal_successes
        
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(subgoal_successes)
        info['num_subgoal_in_focus_success_steps'] = self.num_subgoal_in_focus_success_steps # number of steps the current subgoal in focus has been successfully achieved
        # episode is done also if the current subgoal we are focusing on is achieved
        done = self.subgoal_successes[self.curr_subgoal.pddl_repr()]

        if self.args.termination_type=='no_termination':
            done = False # turn off the done flag if not terminating on success
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode
        elif self.args.termination_type == 'on_non_terminal_subgoal_success':
            # only turn off the done flag if the current subgoal in focus in the last subgoal in the operator
            effects = remove_duplicate_effects(self.grounded_operator)
            if self.curr_subgoal.pddl_repr() == effects[-1].pddl_repr():
                done = False
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode
        elif self.args.termination_type == 'on_first_subgoal_success':
            # only keep the done flag if the current subgoal in focus is the first subgoal in the operator
            effects = remove_duplicate_effects(self.grounded_operator)
            if self.curr_subgoal.pddl_repr() != effects[0].pddl_repr():
                done = False
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode

        # save info to the csv file, one row per each collision point
        # save every 10 episodes every 10 timesteps
        if self.record_rollouts and self.time_step % 100 == 0:
            for collision_point, penalty in zip(collision_points, penalties):
                g_pos = obs_with_semantics['gripper1_pos']
                self.csv_writer.writerow([g_pos[0], g_pos[1], g_pos[2], collision_point[0], collision_point[1], collision_point[2], penalty, reward, sum(list(self.subgoal_successes.values())), reward + sum(penalties) == 0, self.time_step, self.episode])
            self.csv_file.flush()
        
        self.time_step += 1
        return obs, reward + sum(penalties), done, truncated, info

    def reset(self, **kwargs):
        reset_success = False
        while not reset_success:
            # first, reset the environment to the very beginning
            try: # kwargs include the seed
                obs, info = self.env.reset(**kwargs)
            except:
                obs = self.env.reset(**kwargs)
                info = {}
            # second, execute the executors that should be executed before the operator to learn
            reset_success = True
            for op, ex in self.executed_operators.items():
                ex_success = ex.execute(self.detector, op)
                if not ex_success:
                    reset_success = False
                    break
            if len(self.executed_operators) > 0:
                print("all executors executed successfully")
        self.unwrapped.timestep=0 # reset the timesteps
        # reset the episode rewards and penalties
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        # reset the success of the subgoals
        self.subgoal_successes, self.episode_historical_subgoal_successes, _ = self._init_subgoal_dicts()
        self.detector.update_obs()
        self.episode += 1
        # reset the infos
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = self.subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        subgoal_successes = []
        for subgoal, success in self.subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
            subgoal_successes.append(success)
        info['goal_progress'] = subgoal_successes

        # save additional subgoal historical success info for each subgoal
        historical_subgoal_successes = []
        for subgoal, historical_success in self.episode_historical_subgoal_successes.items():
            info[f'{subgoal}_historical_subgoal'] = historical_success
            historical_subgoal_successes.append(historical_success)
        info['historical_subgoal_successes'] = historical_subgoal_successes
        
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(self.subgoal_successes.values())
        info['is_success'] = info['goal_success'] # for compatibility with the stable_baselines3's update_info_buffer
        info['episode'] = { # for compatibility with the stable_baselines3's update_info_buffer
            'ep_cumu_r_shaping': self.episode_r_shaping,
            'ep_cumu_col_penalty': self.episode_collision_penalty,
            'ep_cumu_collisions': self.episode_num_collisions,
            'subgoal_success': self.subgoal_successes[self.curr_subgoal.pddl_repr()],
        }
        self.num_subgoal_in_focus_success_steps = 0 # reset the number of steps the current subgoal in focus has been successfully achieved
        # reset the time step
        info['num_subgoal_in_focus_success_steps'] = self.num_subgoal_in_focus_success_steps # number of steps the current subgoal in focus has been successfully achieved
        self.time_step = 0
        return obs, info
    
    
    def reward(self, reward:float):
        """reward the agent with a reward. Overwrites the reward function in the environment. Duplicates part of the logic in :meth:`step` to compute the reward based on the observation with semantics in case the reward function is called outside of the step function by the learning algorithm

        Args:
            reward (float): the reward to give to the agent
        """
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        penalties, _= self.collision_penalty(obs_with_semantics)
        return reward + sum(penalties)

    def compute_reward(self, prev_numeric_obs_with_semantics:dict, current_numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            prev_numeric_obs_with_semantics: the observation before taking the action in which the keys have semantics and the values are arrays of numeric values
            current_numeric_obs_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
            binary_obs: the nerw binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward
        """

        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        effects = remove_duplicate_effects(self.grounded_operator)
        
        subgoal_bonuses = [0]
        
        for i, effect in enumerate(effects):
            max_progress_rw_per_step = i + 1
            max_episode_steps = self.args.ep_len * (i + 1)
            max_prev_subgoal_rw_per_step = subgoal_bonuses[i]
            subgoal_bonuses.append((max_prev_subgoal_rw_per_step + max_progress_rw_per_step) * max_episode_steps)
        
        if 'pos_exp_dist_test' == self.args.rw_type:
            subgoal_bonuses = [0]
            shaping_reward_max = [1]
            for i, effect in enumerate(effects):
                max_episode_steps = self.args.ep_len * (i + 1)
                max_prev_subgoal_rw_per_step = (subgoal_bonuses[i] + shaping_reward_max[i])*max_episode_steps
                subgoal_bonuses.append(max_prev_subgoal_rw_per_step)
                shaping_reward_max.append(max_prev_subgoal_rw_per_step)
        
        if 'vel' in self.args.rw_type:
            subgoal_bonuses = [1]
            
            for i, effect in enumerate(effects):
                max_episode_steps = self.args.ep_len * (i + 1)
                max_prev_subgoal_rw_per_step = subgoal_bonuses[i]
                subgoal_bonuses.append(max_prev_subgoal_rw_per_step* max_episode_steps)
        
        if self.args.termination_type=='no_termination': # not terminating on success means the subgoal bonuses can be smaller
            # we design subgoal bonuses to be powers of 
            subgoal_bonuses = [1]
            for i, effect in enumerate(effects):
                subgoal_bonuses.append(subgoal_bonuses[i] * 5)
        
        if self.args.termination_type in ('on_non_terminal_subgoal_success', 'on_first_subgoal_success'):
            subgoal_bonuses = [1]
            for i, effect in enumerate(effects):
                subgoal_bonuses.append(subgoal_bonuses[i] * 10)

        step_cost = 0
        if 'neg' in self.args.rw_type:
            step_cost = - subgoal_bonuses[-1]
        sub_goal_reward = 0

        # reset subgoal successes
        for effect in self.subgoal_successes.keys():
            # set subgoal success to False for all effects that are in the subgoal successes dictionary
            self.subgoal_successes[effect] = False

        for i, effect in enumerate(effects):
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if check_effect_satisfied(effect, self.curr_subgoal, binary_obs, self.grounded_operator):
                self.subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                self.episode_historical_subgoal_successes[effect.pddl_repr()] = True # record if the subgoal was achieved at least once in the episode
                # subgoal bonus increases with the number of subgoals achieved
                sub_goal_reward += subgoal_bonuses[i + 1]
                # if effect is the current subgoal and it is satisfied, return the reward immediately
                if effect.pddl_repr() == self.curr_subgoal.pddl_repr():
                    self.num_subgoal_in_focus_success_steps += 1 # increment the number of steps the current subgoal in focus has been successfully achieved
                    break
            else:
                self.subgoal_successes[effect.pddl_repr()] = False
                llm_reward_shaping_fn = self.subgoal_reward_shaping_fn_mapping.get(effect.pddl_repr())
                if llm_reward_shaping_fn is None:
                    print(f"LLM reward shaping function for the effect {effect.pddl_repr()} not set")
                else:
                    try: # the llm reward shaping function may not be error-free. In that case, print error
                        if 'pos_exp_dist' == self.args.rw_type and not self.args.termination_type=='no_termination':
                            sub_goal_reward += subgoal_bonuses[i] * llm_reward_shaping_fn(current_numeric_obs_with_semantics, f"({effect.pddl_repr()})") # the reward shaping function should return a reward in the range of [0, 1].
                        elif 'pos_exp_dist' == self.args.rw_type:
                            sub_goal_reward += (i + 1) * llm_reward_shaping_fn(current_numeric_obs_with_semantics, f"({effect.pddl_repr()})") # the reward shaping function should return a reward in the range of [0, 1]. 
                        elif 'pos_exp_dist_test' == self.args.rw_type:
                            sub_goal_reward += shaping_reward_max[i] * llm_reward_shaping_fn(current_numeric_obs_with_semantics, f"({effect.pddl_repr()})")
                        elif 'vel' in self.args.rw_type:
                            sub_goal_reward += subgoal_bonuses[i] * llm_reward_shaping_fn(prev_numeric_obs_with_semantics, current_numeric_obs_with_semantics, f"({effect.pddl_repr()})") # the reward shaping function should return a reward in the range of [0, 1].
                    except Exception as e:
                        raise(f"Error in the LLM reward shaping function for the effect {effect.pddl_repr()}: {e}")
                break # return the reward as soon as one effect is not satisfied. Assume later effects are at 0% progress therefore would get a shaping reward of 0 anyway.
        
        return step_cost + sub_goal_reward
    
    def collision_penalty(self, numeric_obs_with_semantics:dict) -> Tuple[float, List[np.array]]:
        """Penalize the robot for getting too close to objects it is not supposed to collide with

        Args:
            numeric_obs_with_semantics: the observation in which the keys have semantics and the values are arrays of numeric values

        Returns:
            penalties: a list of penalties for getting too close to objects
            collision_points: a list of 3D collision points
        """
        # find objects that the robot is allowed to collide with. These are objects that the robot grasps either in the precondition or the effects of the grounded operator
        allowed_objects = []
        collision_threshold = 0.01 # getting closer than this distance will incur a penalty
        for effect in self.grounded_operator.effects:
            if effect.atom.predicate.name == 'exclusively-occupying-gripper':
                for arg in effect.atom.subterms:
                    # find the parameter that's not the gripper
                    if arg.name != 'gripper1':
                        allowed_objects.append(arg.name)
        for condition in self.grounded_operator.precondition.subformulas:
            if not hasattr(condition, 'connective') and condition.predicate.name == 'exclusively-occupying-gripper':
                for arg in condition.subterms:
                    # find the parameter that's not the gripper
                    if arg.name != 'gripper1':
                        allowed_objects.append(arg.name)
        # find the objects that the robot is close to
        penalties = []
        collision_points = []
        for key, obs in numeric_obs_with_semantics.items():
            if 'collision_dist' in key and not any(obj in key for obj in allowed_objects):
                if obs[0] <= collision_threshold: # the first element is the collision distance.
                    penalties.append(-1/(obs[0]+0.001)) # the closer the robot gets to the object, the higher the penalty. Add a small value to avoid division by zero
                    collision_points.append(obs[1:]) # the rest of the elements are the 3D collision point coordinates
        return penalties, collision_points
    
    def close(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
        return super().close()
    
    def _discretize_gripper_action(self, action:np.array) -> np.array:
        """discretize the gripper opening action into 3 discrete actions: open, close, and do nothing

        Args:
            action (np.array): the action to discretize

        Returns:
            np.array: the discretized action
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        gripper_opening_min = self.env.action_space.low[-1]
        gripper_opening_max = self.env.action_space.high[-1]
        gripper_opening_range = gripper_opening_max - gripper_opening_min
        gripper_close_threshold = gripper_opening_min + gripper_opening_range/3
        gripper_open_threshold = gripper_opening_max - gripper_opening_range/3
        if action[-1] < gripper_close_threshold: # close the gripper
            action[-1] = gripper_opening_min
        elif action[-1] > gripper_open_threshold: # open the gripper
            action[-1] = gripper_opening_max
        else: # do nothing
            action[-1] = 0
        return action
    
class BaseOperatorWrapper(OperatorWrapper):
    def __init__(self, env, config, args, grounded_operator, record_rollouts = False):
        # set the subgoal as the last effect of the grounded operator
        effects = remove_duplicate_effects(grounded_operator)
        curr_subgoal = effects[-1]
        super().__init__(env, config, args, grounded_operator, curr_subgoal, record_rollouts)
        self.reward_range = (-1, 0)
    
    def compute_reward(self, prev_numeric_obs_with_semantics:dict, current_numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            prev_numeric_obs_with_semantics: the observation before taking the action in which the keys have semantics and the values are arrays of numeric values
            current_numeric_obs_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
            binary_obs: the nerw binary observation whose keys are predicates and values are True/False
        Returns:
            float: the reward between [-1, 0]
        """
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        effects = remove_duplicate_effects(self.grounded_operator)
        
        if self.args.termination_type=='on_all_subgoal_success': # if the episode is terminated on success, the completion bonus is 0
            completion_bonus = self.args.ep_len
        elif self.args.termination_type in ('on_first_subgoal_success',  'on_non_terminal_subgoal_success'): # if the episode is not terminated on success, the completion bonus is 10
            completion_bonus = 10
        else:
            completion_bonus = 5
        step_cost = 0
        if 'step_cost' in self.args.rw_type:
            step_cost = -1


        # reset subgoal successes
        for effect in self.subgoal_successes.keys():
            # set subgoal success to False for all effects that are in the subgoal successes dictionary
            self.subgoal_successes[effect] = False

        for _, effect in enumerate(effects):
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if check_effect_satisfied(effect, self.curr_subgoal, binary_obs, self.grounded_operator):
                self.subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                self.episode_historical_subgoal_successes[effect.pddl_repr()] = True # record if the subgoal was achieved at least once in the episode
                # subgoal bonus increases with the number of subgoals achieved
                # if effect is the current subgoal and it is satisfied, return the reward immediately
                if effect.pddl_repr() == self.curr_subgoal.pddl_repr():
                    self.num_subgoal_in_focus_success_steps += 1 # increment the number of steps the current subgoal in focus has been successfully achieved
                    break
            else:
                self.subgoal_successes[effect.pddl_repr()] = False
                return step_cost # return the -1 step cost as soon as one effect is not satisfied.
        
        return step_cost + completion_bonus
class LLMAblatedOperatorWrapper(OperatorWrapper):
    def __init__(self, env:MujocoEnv, config:dict, args:argparse.Namespace, grounded_operator:fs.Action, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env, config, args, grounded_operator, curr_subgoal, record_rollouts)


    def compute_reward(self, prev_numeric_obs_with_semantics:dict, current_numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            prev_numeric_obs_with_semantics: the observation before taking the action in which the keys have semantics and the values are arrays of numeric values
            current_numeric_obs_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
            binary_obs: the nerw binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward between [-1, number of effects - 1]
        """
        
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        duplicate_grasp_effects = check_duplicate_grasp_effects(self.grounded_operator)
        duplicate_ungrasp_effects = check_duplicate_free_gripper_effects(self.grounded_operator)
        num_subgoals = len(effects) -1 if (duplicate_grasp_effects or duplicate_ungrasp_effects) else len(effects)
        # there is a step cost of -1 regardless
        step_cost = -num_subgoals
         
        sub_goal_reward = 0
        
        # reset subgoal successes
        for effect in self.subgoal_successes.keys():
            # set subgoal success to False for all effects that are in the subgoal successes dictionary
            self.subgoal_successes[effect] = False

        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            if effect.pddl_repr() == 'free gripper1' and duplicate_ungrasp_effects: # if the effect is `free gripper1`, skip it since it is the same effect as `not (exclusively-occupying-gripper ?object gripper1)`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if check_effect_satisfied(effect, self.curr_subgoal, binary_obs, self.grounded_operator):
                self.subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                self.episode_historical_subgoal_successes[effect.pddl_repr()] = True # record if the subgoal was achieved at least once in the episode
                sub_goal_reward += 1
                # if effect is the current subgoal and it is satisfied, return the reward immediately
                if effect.pddl_repr() == self.curr_subgoal.pddl_repr():
                    self.num_subgoal_in_focus_success_steps += 1 # increment the number of steps the current subgoal in focus has been successfully achieved
                    break
            else:
                self.subgoal_successes[effect.pddl_repr()] = False
                break # return the reward as soon as one effect is not satisfied. Assume later effects are not satisfied.
        
        return step_cost + sub_goal_reward
    

class CollisionAblatedOperatorWrapper(OperatorWrapper):
    def __init__(self, env:MujocoEnv, config:dict, args:argparse.Namespace, grounded_operator:fs.Action, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env, config, args, grounded_operator, curr_subgoal, record_rollouts)
        self.reward_range = (-1, 0)

    def reward(self, reward:float):
        """reward the agent with a reward. Overwrites the reward function in the environment. Duplicates part of the logic in :meth:`step` to compute the reward based on the observation with semantics in case the reward function is called outside of the step function by the learning algorithm

        Args:
            reward (float): the reward to give to the agent
        """
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        return reward
    
    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """step function that steps the environment and computes the reward based on the observation with semantics

        Args:
            action (array): 7 dimensional array representing the action to take

        Returns:
            Tuple[np.array, float, bool, bool, dict]: the observation, reward, done, truncated, and info
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        action = self._discretize_gripper_action(action)
        # get the observation with semantics before taking the action, make sure it does not get modified by the step function
        prev_obs_with_semantics:dict = self.detector.get_obs()
        # preserve the values of the dictionary so they are not modified by the step function
        prev_obs_with_semantics = {k: v.copy() for k, v in prev_obs_with_semantics.items()}
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env.unwrapped)
        reward = self.compute_reward(prev_numeric_obs_with_semantics=prev_obs_with_semantics, current_numeric_obs_with_semantics=obs_with_semantics, binary_obs=binary_obs)
        # save reward and penalties separately
        self.episode_r_shaping += reward
        self.episode_collision_penalty += 0 # assume no collision
        self.episode_num_collisions += 0 # assume no collision
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        
        # save subgoal successes
        info['subgoal_success'] = self.subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        subgoal_successes = []
        for subgoal, success in self.subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
            subgoal_successes.append(success)
        # save additional subgoal historical success info for each subgoal
        historical_subgoal_successes = []
        for subgoal, historical_success in self.episode_historical_subgoal_successes.items():
            info[f'{subgoal}_historical_subgoal'] = historical_success
            historical_subgoal_successes.append(historical_success)
        info['historical_subgoal_successes'] = historical_subgoal_successes

        info['goal_progress'] = subgoal_successes
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(self.subgoal_successes.values())
        # episode is done also if the current subgoal we are focusing on is achieved
        info['num_subgoal_in_focus_success_steps'] = self.num_subgoal_in_focus_success_steps # number of steps the current subgoal in focus has been successfully achieved
        done = self.subgoal_successes[self.curr_subgoal.pddl_repr()]

        if self.args.termination_type=='no_termination':
            done = False # turn off the done flag if not terminating on success
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode
        elif self.args.termination_type == 'on_non_terminal_subgoal_success':
            # only turn off the done flag if the current subgoal in focus in the last subgoal in the operator
            effects = remove_duplicate_effects(self.grounded_operator)
            if self.curr_subgoal.pddl_repr() == effects[-1].pddl_repr():
                done = False
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode
        elif self.args.termination_type == 'on_first_subgoal_success':
            # only keep the done flag if the current subgoal in focus is the first subgoal in the operator
            effects = remove_duplicate_effects(self.grounded_operator)
            if self.curr_subgoal.pddl_repr() != effects[0].pddl_repr():
                done = False
            info['subgoal_success'] = self.episode_historical_subgoal_successes[self.curr_subgoal.pddl_repr()] # use the episode subgoal success instead of the current action step subgoal success
            info['goal_progress'] = historical_subgoal_successes # use the episode subgoal successes instead of the current action step subgoal successes
            info['goal_success'] = all(historical_subgoal_successes) # overall goal success is if all subgoals are achieved in the episode

        self.time_step += 1
        return obs, reward, done, truncated, info

class CollisionLLMAblatedOperatorWrapper(CollisionAblatedOperatorWrapper):
    def __init__(self, env:MujocoEnv, config:dict, args:argparse.Namespace, grounded_operator:fs.Action, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env, config, args, grounded_operator, curr_subgoal, record_rollouts)
        self.reward_range = (-1, 0)

    def compute_reward(self, prev_numeric_obs_with_semantics:dict, current_numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            prev_numeric_obs_with_semantics: the observation before taking the action in which the keys have semantics and the values are arrays of numeric values
            current_numeric_obs_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
            binary_obs: the nerw binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward between [-1, number of effects - 1]
        """
        
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        effects = remove_duplicate_effects(self.grounded_operator)
        
        subgoal_bonuses = [0]
        for i, effect in enumerate(effects):
            max_progress_rw_per_step = i + 1
            max_episode_steps = self.args.ep_len * (i + 1)
            max_prev_subgoal_rw_per_step = subgoal_bonuses[i]
            subgoal_bonuses.append((max_prev_subgoal_rw_per_step + max_progress_rw_per_step) * max_episode_steps)
        
        if 'vel' in self.args.rw_type:
            subgoal_bonuses = [1]
            
            for i, effect in enumerate(effects):
                max_episode_steps = self.args.ep_len * (i + 1)
                max_prev_subgoal_rw_per_step = subgoal_bonuses[i]
                subgoal_bonuses.append(max_prev_subgoal_rw_per_step* max_episode_steps)
        
        if self.args.termination_type=='no_termination': # not terminating on success means the subgoal bonuses can be smaller
            # we design subgoal bonuses to be powers of 
            subgoal_bonuses = [1]
            for i, effect in enumerate(effects):
                subgoal_bonuses.append(subgoal_bonuses[i] * 5)
        
        if self.args.termination_type in ('on_non_terminal_subgoal_success', 'on_first_subgoal_success'):
            subgoal_bonuses = [1]
            for i, effect in enumerate(effects):
                subgoal_bonuses.append(subgoal_bonuses[i] * 10)
               
        step_cost = 0
        if 'step_cost' in self.args.rw_type:
            step_cost = -1
        sub_goal_reward = 0

        # reset subgoal successes
        for effect in self.subgoal_successes.keys():
            # set subgoal success to False for all effects that are in the subgoal successes dictionary
            self.subgoal_successes[effect] = False

        for i, effect in enumerate(effects):
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if check_effect_satisfied(effect, self.curr_subgoal, binary_obs, self.grounded_operator):
                self.subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                self.episode_historical_subgoal_successes[effect.pddl_repr()] = True # record if the subgoal was achieved at least once in the episode
                # subgoal bonus increases with the number of subgoals achieved
                sub_goal_reward += subgoal_bonuses[i + 1]
                # if effect is the current subgoal and it is satisfied, return the reward immediately
                if effect.pddl_repr() == self.curr_subgoal.pddl_repr():
                    self.num_subgoal_in_focus_success_steps += 1 # increment the number of steps the current subgoal in focus has been successfully achieved
                    break
            else:
                self.subgoal_successes[effect.pddl_repr()] = False
                break # return the reward as soon as one effect is not satisfied. Assume later effects are at 0% progress therefore would get a shaping reward of 0 anyway.
        
        return step_cost + sub_goal_reward