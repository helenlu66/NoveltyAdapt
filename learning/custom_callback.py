import copy
import time
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import csv
import stable_baselines3
import logging
from tarski import fstrips as fs
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import *
from utils import *

class RenderCallback(EventCallback):
    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
    
    def _on_step(self):
        self.training_env.render_mode = self.render_mode
        self.training_env.render()
        return True

class ProgressRemainingCallback(EventCallback):
    """
    Callback that allows to change the learning rate of the model during training.
    """

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.max_timesteps = total_timesteps

    def on_rollout_end(self) -> bool:
        progress_remaining = 1 - (self.model.num_timesteps / self.max_timesteps)
        # set the real progress remaining
        self.model._current_progress_remaining = progress_remaining
        return True

class TerminateOnOperatorDiscoveryCallback(BaseCallback):
    def __init__(self, op_save_path, recent_model_save_path, recent_model_buffer_path, verbose=1):
        super(TerminateOnOperatorDiscoveryCallback, self).__init__(verbose)
        self.discovered_operator = None
        self.save_path = op_save_path
        self.save_freq = self.num_timesteps / 50
        self.recent_model_save_path = recent_model_save_path

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0:
            if self.recent_model_save_path is not None:
                self.model.save(os.path.join(self.recent_model_save_path, "recent_model"))
                # check if the model is off-policy and save the replay buffer
                if hasattr(self.model, 'save_replay_buffer'):
                    self.model.save_replay_buffer(self.recent_model_buffer_path)

        info = self.locals.get("infos", {})[0]
        discovered_op = info.get("discovered_operator")
        if discovered_op is not None:
            self.discovered_operator = discovered_op
            if self.verbose > 0:
                print(f"Discovered operator: {self.discovered_operator}")
            # Save the discovered operator to a file at save_path + "/discovered_operator.pkl"
            with open(os.path.join(self.save_path, "discovered_operator.json"), "w") as f:
                # clean the operator dict
                clean_dict = {k:bool(v) for k, v in self.discovered_operator.items() if v is not None}
                json.dump(clean_dict, f, indent=4)
            # Save the changed symbol trace
            changed_symbol_trace = info.get("changed_symbols", [])
            # clean the changed_symbol_trace
            changed_symbol_trace = [{k:bool(v) for k, v in d.items() if v is not None} for d in changed_symbol_trace]
            with open(os.path.join(self.save_path, "changed_symbol_trace.json"), "w") as f:
                json.dump(changed_symbol_trace, f, indent=4)
            return False
        return True  # continue training if no operator was discovered

class SubgoalSuccessEvalCallback(EvalCallback):
    def __init__(self, subgoal_idx, eval_env, logger, best_model_save_path, log_path, best_model_buffer_path = None, recent_model_save_path=None, recent_model_buffer_path=None, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, render_mode='human', verbose=1, early_stopping=False):
        super().__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.early_stopping = early_stopping
        self.subgoal_in_focus_idx = subgoal_idx
        self.best_subgoal_in_focus_success_rate = 0
        self.evaluations_best_subgoal_in_focus_success_rate: List[float] = []
        self.best_goal_progress = 0
        self.evaluations_best_goal_progress: List[float] = []
        
        self.eval_env.render_mode = render_mode
        self.best_model_buffer_path = best_model_buffer_path if best_model_buffer_path is not None else os.path.join(best_model_save_path, "buffer.pkl")
        self.recent_model_save_path = recent_model_save_path
        self.recent_model_buffer_path = recent_model_buffer_path if recent_model_buffer_path is not None else os.path.join(recent_model_save_path, "buffer.pkl")
        self.custom_logger = logger
        self._subgoal_in_focus_successes_buffer: List[bool] = [] # stores the per episode subgoal successes for the subgoal we are currently training for
        self.evaluations_subgoal_in_focus_successes: List[List[bool]] = [] # stores the subgoal successes for each round of evaluation i.e a few episodes per evaluation for the subgoal we are currently training for
        self.evaluations_subgoal_successes: List[List[List[bool]]] = [] # stores the subgoals' successes for each round of evaluation i.e a few episodes per evaluation for all subgoals
        self._historical_subgoal_successes: List[List[bool]] = [] # stores the historical successes (has it ever been achieved at least once per episode) 
        self.evaluations_episode_historical_subgoal_successes: List[List[List[bool]]] = [] # stores the subgoals' historical successes (has it ever been achieved at least once per episode) for each round of evaluation i.e a few episodes per evaluation for all subgoals
        self._subgoal_successes: List[List[bool]] = [] # stores the per episode subgoals' successes. Each episode has a list of booleans indicating the success of each subgoal
        self.evaluations_goal_progress: List[List[float]] = [] # stores the percent progress made towards the goal for each round of evaluation i.e a few episodes per evaluation
        self._num_subgoal_in_focus_success_steps_buffer: List[int] = [] # stores the number of subgoal in focus successes per episode
        self.evaluations_num_subgoal_in_focus_success_steps: List[List[int]] = [] # stores the number of subgoal in focus successes for each round of evaluation i.e a few episodes per evaluation

        self._ep_r_shaping_buffer: List[float] = [] # stores the per episode reward shaping values
        self.evaluations_r_shaping: List[List[float]] = [] # stores the reward shaping values for each round of evaluation i.e a few episodes per evaluation
        self._recover_evaluation_records()
    
    def get_recent_subgoal_success_rate(self):
        """Return the recent subgoal success rate
        """
        if len(self.evaluations_subgoal_successes) == 0:
            return 0
        return np.mean(self.evaluations_subgoal_in_focus_successes[-1])
    
    def get_recent_ave_subgoal_success_rate(self):
        """Return the average subgoal success rate of the most recent 5 evaluations
        """
        # if there are less than 5 evaluations, return the average of all evaluations
        if len(self.evaluations_subgoal_in_focus_successes) < 5:
            mean_success_rate_per_evaluation = np.mean(self.evaluations_subgoal_in_focus_successes, axis=1)
            return np.mean(mean_success_rate_per_evaluation)
        # return the average of the most recent 5 evaluations
        mean_success_rate_per_evaluation = np.mean(self.evaluations_subgoal_in_focus_successes[-5:], axis=1)
        return np.mean(mean_success_rate_per_evaluation)

    def get_best_subgoal_in_focus_success_rate(self):
        """Return the best subgoal success rate
        """
        return self.best_subgoal_in_focus_success_rate
    
    def predict_next_subgoal_success_rate(self):
        """Predict the next subgoal success rate for the subgoal in focus
        """
        # first calculate the slope of the last two subgoal success rates
        if len(self.evaluations_subgoal_in_focus_successes) < 2:
            return self.get_recent_subgoal_success_rate()
        recent_successes = self.evaluations_subgoal_in_focus_successes[-2:]
        recent_success_rates = np.mean(recent_successes, axis=1)
        slope = recent_success_rates[-1] - recent_success_rates[-2]
        return recent_success_rates[-1] + slope
    
    def check_early_stopping(self, min_num_evaluations:int=5, mean_success_rate_threshold:float=0.9, std_threshold:float=0.1) -> Tuple[bool, np.ndarray, float]:
        """Check if the early stopping criteria are satisfied based on the evaluations of the subgoal in focus.
        The criteria are:
        1. The minimum number of evaluations is met.
        2. The mean success rate is above the threshold.
        3. The standard deviation of the success rate is below the threshold.

        Args:
            min_num_evaluations (int): the minimum number of evaluations required
            mean_success_rate_threshold (float): the threshold for the mean success rate
            std_threshold (float): the threshold for the standard deviation of the success rate
        Returns:
            bool: True if the early stopping criteria are satisfied, False otherwise
        """
        if len(self.evaluations_subgoal_in_focus_successes) < min_num_evaluations:
            return False, None, None
        # calculate the mean success rate
        success_rate_per_eval = np.mean(self.evaluations_subgoal_in_focus_successes[-min_num_evaluations:], axis=1)
        std_success_rate_per_evaluation = np.std(success_rate_per_eval)
        mean_success_rate_per_eval = np.mean(success_rate_per_eval)
        if mean_success_rate_per_eval >= mean_success_rate_threshold and std_success_rate_per_evaluation <= std_threshold:
            return True, success_rate_per_eval, std_success_rate_per_evaluation
        return False, success_rate_per_eval, std_success_rate_per_evaluation
    
    def _recover_evaluation_records(self):
        """Recover the evaluation records from the evaluations.npz file
        """
        if os.path.exists(f"{self.log_path}.npz"):
            data = np.load(f"{self.log_path}.npz")
            self.evaluations_timesteps = list(data['timesteps'])
            self.evaluations_results = list(data['results'])
            self.evaluations_length = list(data['ep_lengths'])
            self.evaluations_goal_progress = list(data['goal_progress'])
            self.evaluations_subgoal_successes = list(data['subgoal_successes'])
            # check if 'r_shaping' is in the data
            if 'r_shaping' in data:
                self.evaluations_r_shaping = list(data['r_shaping'])
            else:
                self.evaluations_r_shaping = list(data['results']) # old evaluations files did not have r_shaping
            # check if 'goal_successes' is in the data. This is the overall goal success rate
            if 'goal_successes' in data:
                self.evaluations_successes = list(data['goal_successes'])
            else:
                # deduce the goal success rate from goal progress
                self.evaluations_successes = list(data['successes'])
            # check if 'subgoal_in_focus_successes' is in the data
            if 'subgoal_in_focus_successes' in data:
                self.evaluations_subgoal_in_focus_successes = list(data['subgoal_in_focus_successes'])
            else:
                # deduce the subgoal in focus success rate from subgoal successes
                subgoal_successes = data['subgoal_successes']
                self.evaluations_subgoal_in_focus_successes = list(subgoal_successes[:, :, self.subgoal_in_focus_idx])
            # check if 'historical_subgoal_successes' is in the data
            if 'historical_subgoal_successes' in data:
                self.evaluations_episode_historical_subgoal_successes = list(data['historical_subgoal_successes'])
            if 'best_subgoal_success_rate_so_far' in data:
                self.evaluations_best_subgoal_in_focus_success_rate = list(data['best_subgoal_success_rate_so_far'])
            if 'best_goal_progress_so_far' in data:
                self.evaluations_best_goal_progress = list(data['best_goal_progress_so_far'])
            if 'num_subgoal_in_focus_success_steps' in data:
                self.evaluations_num_subgoal_in_focus_success_steps = list(data['num_subgoal_in_focus_success_steps'])
            else:
                # fill the num_subgoal_in_focus_success_steps with none
                self.evaluations_num_subgoal_in_focus_success_steps = [[None]*len(self.evaluations_subgoal_successes[0])] * len(self.evaluations_subgoal_successes)
            
    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset buffers
            self._is_success_buffer = []
            self._subgoal_in_focus_successes_buffer = []
            self._ep_r_shaping_buffer = []
            self._subgoal_successes = []
            self._historical_subgoal_successes = []
            self._num_subgoal_in_focus_success_steps_buffer = []
            print(f"evaluating model at {self.recent_model_save_path}")
            episode_rewards, episode_lengths = evaluate_policy( # this evaluates the model for a few policies
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_subgoal_success_callback,
            )


            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            # Log the evaluation values
            success_rate = 0
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean Success rate per episode: {100 * success_rate:.2f}%")
            self.logger.record("eval/mean_goal_success_rate_per_ep", success_rate)
            
            subgoal_in_focus_success_rate = 0
            if len(self._subgoal_in_focus_successes_buffer) > 0:
                subgoal_in_focus_success_rate = np.mean(self._subgoal_in_focus_successes_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean in-focus subgoal success rate per episode: {100 * subgoal_in_focus_success_rate:.2f}%")
            self.logger.record("eval/mean_ep_subgoal_in_focus_success_rate", subgoal_in_focus_success_rate)

            subgoal_success_rates = None
            if len(self._subgoal_successes) > 0:
                subgoal_success_rates = np.mean(self._subgoal_successes, axis=0)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean subgoals success rates per episode: {100 * subgoal_success_rates}")
            self.logger.record("eval/mean_ep_subgoal_success_rates", subgoal_success_rates)

            mean_historical_progress_to_goal = 0
            if len(self._historical_subgoal_successes) > 0:
                progress_to_goal:List = np.mean(self._historical_subgoal_successes, axis=1)
                mean_historical_progress_to_goal = np.mean(progress_to_goal) # average across episodes
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean historical progress to goal per episode: {100 * mean_historical_progress_to_goal:.2f}%")
            self.logger.record("eval/mean_historical_progress_to_goal", mean_historical_progress_to_goal)

            mean_progress_to_goal_at_termination = 0
            if len(self._historical_subgoal_successes) > 0:
                progress_to_goal:List = np.mean(self._subgoal_successes, axis=1)
                mean_progress_to_goal_at_termination = np.mean(progress_to_goal) # average across episodes
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean progress to goal per episode at termination: {100 * mean_historical_progress_to_goal:.2f}%")
            self.logger.record("eval/mean_progress_to_goal_at_termination", mean_progress_to_goal_at_termination)

            historical_subgoal_success_rates = 0
            if len(self._historical_subgoal_successes) > 0:
                historical_subgoal_success_rates = np.mean(self._historical_subgoal_successes, axis=0)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean historical subgoals success rates per episode: {100 * historical_subgoal_success_rates}")
            self.logger.record("eval/mean_ep_historical_subgoal_success_rates", historical_subgoal_success_rates)
            
            num_subgoal_in_focus_success_steps = 0
            if len(self._num_subgoal_in_focus_success_steps_buffer) > 0:
                num_subgoal_in_focus_success_steps = np.mean(self._num_subgoal_in_focus_success_steps_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean number of subgoal in focus successes per episode: {num_subgoal_in_focus_success_steps:.2f}")
            self.logger.record("eval/mean_num_subgoal_in_focus_success_steps", num_subgoal_in_focus_success_steps)

            mean_ep_r_shaping = -float('inf')
            if len(self._ep_r_shaping_buffer) > 0:
                mean_ep_r_shaping = np.mean(self._ep_r_shaping_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode reward shaping per episode: {mean_ep_r_shaping:.2f}")
            self.logger.record("eval/mean_ep_r_shaping", mean_ep_r_shaping)


            # Dump log so the evaluation results are printed with the correct timestep
            if self.verbose > 0:
                self.custom_logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"mean episode reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # get the min and max episode reward too
                self.custom_logger.info(f"Min episode reward: {np.min(episode_rewards)}, " f"Max episode reward: {np.max(episode_rewards)}")
                self.custom_logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                self.custom_logger.info(f"Min episode length: {np.min(episode_lengths)}, " f"Max episode length: {np.max(episode_lengths)}")
            # Add to current Logger
            self.logger.record("eval/mean_ep_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)           
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward >= self.best_mean_reward:
                if self.verbose > 0:
                    self.custom_logger.info("New best mean reward!")
                if subgoal_in_focus_success_rate >= self.best_subgoal_in_focus_success_rate:
                    if self.verbose > 0:
                        self.custom_logger.info("New best success rate!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                        # check if the model is off-policy and save the replay buffer
                        if hasattr(self.model, 'save_replay_buffer'):
                            self.model.save_replay_buffer(self.best_model_buffer_path)
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                    self.best_subgoal_in_focus_success_rate = subgoal_in_focus_success_rate
                    self.best_mean_reward = mean_reward
                    
            
            if self.recent_model_save_path is not None:
                self.model.save(os.path.join(self.recent_model_save_path, "recent_model"))
                # check if the model is off-policy and save the replay buffer
                if hasattr(self.model, 'save_replay_buffer'):
                    self.model.save_replay_buffer(self.recent_model_buffer_path)
            # Save the results in a csv file located in the second to last directory of log_path
            # Split the log_path to get the second to last directory
            csv_path = os.path.split(self.log_path)[0]
            # if results_eval.csv does not exist, create it and write the header
            if not os.path.exists(os.path.join(csv_path, 'results_eval.csv')):
                with open(os.path.join(csv_path, 'results_eval.csv'), 'w') as f:
                    header = "timesteps,num_subgoal_in_focus_success_steps, subgoal_in_focus_success_rate, best_subgoal_in_focus_success_so_far,best_progress_so_far,mean_historical_progress_to_goal,mean_progress_to_goal_at_termination, mean_ep_length,mean_ep_r_shaping,mean_reward\n"
                    f.write(header)
                    f.close()
            with open(os.path.join(csv_path, 'results_eval.csv'), 'a') as f:
                # if file already exists, append the results
                f.write("{},{},{},{},{},{},{},{},{}\n".format(
                    self.num_timesteps,
                    num_subgoal_in_focus_success_steps,
                    subgoal_in_focus_success_rate,
                    self.best_subgoal_in_focus_success_rate,
                    self.best_goal_progress,
                    mean_historical_progress_to_goal,
                    mean_progress_to_goal_at_termination,
                    mean_ep_length,
                    mean_ep_r_shaping,
                    mean_reward 
                ))
                f.close()

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_r_shaping.append(self._ep_r_shaping_buffer)
                self.evaluations_best_goal_progress.append(self.best_goal_progress)
                self.evaluations_best_subgoal_in_focus_success_rate.append(self.best_subgoal_in_focus_success_rate)
                self.best_goal_progress = max(self.best_goal_progress, mean_historical_progress_to_goal)

                kwargs = {}
                kwargs['r_shaping'] = self.evaluations_r_shaping
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs['goal_successes'] = self.evaluations_successes
                if len(self._subgoal_in_focus_successes_buffer) > 0:
                    self.evaluations_subgoal_in_focus_successes.append(self._subgoal_in_focus_successes_buffer)
                    kwargs['subgoal_in_focus_successes'] = self.evaluations_subgoal_in_focus_successes
                if len(self._subgoal_successes) > 0:
                    self.evaluations_subgoal_successes.append(self._subgoal_successes)
                    kwargs['subgoal_successes'] = self.evaluations_subgoal_successes
                if len(self._historical_subgoal_successes) > 0:
                    progress:List[float] = np.mean(self._historical_subgoal_successes, axis=1)
                    self.evaluations_goal_progress.append(progress)
                    kwargs['goal_progress'] = self.evaluations_goal_progress
                    self.evaluations_episode_historical_subgoal_successes.append(self._historical_subgoal_successes)
                    kwargs['historical_subgoal_successes'] = self.evaluations_episode_historical_subgoal_successes
                if len(self._num_subgoal_in_focus_success_steps_buffer) > 0:
                    self.evaluations_num_subgoal_in_focus_success_steps.append(self._num_subgoal_in_focus_success_steps_buffer)
                    kwargs['num_subgoal_in_focus_success_steps'] = self.evaluations_num_subgoal_in_focus_success_steps

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    best_subgoal_success_rate_so_far=self.evaluations_best_subgoal_in_focus_success_rate,
                    best_goal_progress_so_far=self.evaluations_best_goal_progress,
                    **kwargs,
                )

            
            # Early stopping
            # if the subgoal_in_focus_success_rate averages higher than 90% for the past 5 evals and the std dev < 10%
            if self.early_stopping and len(self.evaluations_subgoal_in_focus_successes) >= 5:
                stopped, mean_success_rate_per_evaluation, std_success_rate_per_evaluation = self.check_early_stopping(mean_success_rate_threshold=0.9, std_threshold=0.1)
                if stopped:
                    self.custom_logger.info(f"Early stopping at {self.num_timesteps} timesteps")
                    print(f"Early stopping at {self.num_timesteps} timesteps, with mean success rate {np.mean(mean_success_rate_per_evaluation)} and std dev {std_success_rate_per_evaluation}")
                    self.custom_logger.info(f"Mean subgoal success rate: {np.mean(mean_success_rate_per_evaluation)}")
                    self.custom_logger.info(f"Std subgoal success rate: {np.mean(std_success_rate_per_evaluation)}")
                    continue_training = False

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()


        return continue_training
    
    def _log_subgoal_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the subgoal success rate during evaluation.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if locals_["done"]:
            # store the episode reward shaping values and collision penalties
            self._ep_r_shaping_buffer.append(info.get('ep_cumu_r_shaping'))
            maybe_is_success = info.get("goal_success") # the success of the overall goal
            subgoal_success = info.get("subgoal_success") # the success of the subgoal in focus
            goal_progress = info.get("goal_progress") # an array of booleans indicating the success of each subgoal
            historical_subgoal_successes = info.get("historical_subgoal_successes") # an array of booleans indicating if each subgoal has been achieved at least once per episode
            self._historical_subgoal_successes.append(historical_subgoal_successes)
            #subgoals = [key for key in info.keys() if '_subgoal' in key]
            self._subgoal_in_focus_successes_buffer.append(subgoal_success)
            self._is_success_buffer.append(maybe_is_success)
            self._subgoal_successes.append(goal_progress)
            num_subgoal_in_focus_success_steps = info.get("num_subgoal_in_focus_success_steps", 0)
            self._num_subgoal_in_focus_success_steps_buffer.append(num_subgoal_in_focus_success_steps)


class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, logger, best_model_save_path, log_path, recent_model_save_path=None, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, render_mode='human', verbose=1):
        super().__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.best_subgoal_success_rate = 0
        self.eval_env.render_mode = render_mode
        self.recent_model_save_path = recent_model_save_path
        self.custom_logger = logger
        self._subgoal_successes_buffer: List[bool] = [] # stores the per episode subgoal successes
        self.evaluations_subgoal_successes: List[List[bool]] = [] # stores the subgoal successes for each round of evaluation i.e a few episodes per evaluation
        self._ep_r_shaping_buffer: List[float] = [] # stores the per episode reward shaping values
        self.evaluations_r_shaping: List[List[float]] = [] # stores the reward shaping values for each round of evaluation i.e a few episodes per evaluation
        self._ep_col_penalty_buffer: List[float] = [] # stores the per episode collision penalties
        self.evaluations_col_penalties: List[List[float]] = [] # stores the collision penalties for each round of evaluation i.e a few episodes per evaluation
        self._ep_num_collisions_buffer: List[int] = [] # stores the per episode number of collisions
        self.evaluations_num_collisions: List[List[int]] = [] # stores the number of collisions for each round of evaluation i.e a few episodes per evaluation

    def get_recent_subgoal_success_rate(self):
        """Return the recent subgoal success rate
        """
        if len(self.evaluations_subgoal_successes) == 0:
            return 0
        return np.mean(self.evaluations_subgoal_successes[-1])
        
    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset buffers
            self._is_success_buffer = []
            self._subgoal_successes_buffer = []
            self._ep_r_shaping_buffer = []
            self._ep_col_penalty_buffer = []
            self._ep_num_collisions_buffer = []

            episode_rewards, episode_lengths = evaluate_policy( # this evaluates the model for a few policies
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_subgoal_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_r_shaping.append(self._ep_r_shaping_buffer)
                self.evaluations_col_penalties.append(self._ep_col_penalty_buffer)
                self.evaluations_num_collisions.append(self._ep_num_collisions_buffer)

                kwargs = {}
                kwargs['r_shaping'] = self.evaluations_r_shaping
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)
                if len(self._subgoal_successes_buffer) > 0:
                    self.evaluations_subgoal_successes.append(self._subgoal_successes_buffer)
                    kwargs['subgoal_successes'] = self.evaluations_subgoal_successes
                # save the per episode collision penalties
                if len(self._ep_col_penalty_buffer) > 0:
                    self.evaluations_col_penalties.append(self._ep_col_penalty_buffer)
                    kwargs['col_penalties'] = self.evaluations_col_penalties
                # save the per episode number of collisions
                if len(self._ep_num_collisions_buffer) > 0:
                    self.evaluations_num_collisions.append(self._ep_num_collisions_buffer)
                    kwargs['num_collisions'] = self.evaluations_num_collisions
 

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            # Log the evaluation values
            success_rate = 0
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean Success rate per episode: {100 * success_rate:.2f}%")
            self.logger.record("eval/mean_goal_success_rate_per_ep", success_rate)
            
            subgoal_success_rate = 0
            if len(self._subgoal_successes_buffer) > 0:
                subgoal_success_rate = np.mean(self._subgoal_successes_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean subgoals success rate per episode: {100 * subgoal_success_rate:.2f}%")
            self.logger.record("eval/mean_ep_subgoal_success_rate", subgoal_success_rate)
            
            mean_ep_r_shaping = None
            if len(self._ep_r_shaping_buffer) > 0:
                mean_ep_r_shaping = np.mean(self._ep_r_shaping_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode reward shaping per episode: {mean_ep_r_shaping:.2f}")
            self.logger.record("eval/mean_ep_r_shaping", mean_ep_r_shaping)

            mean_col_penalty = None
            if len(self._ep_col_penalty_buffer) > 0:
                mean_col_penalty = np.mean(self._ep_col_penalty_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode collision penalty per episode: {mean_col_penalty:.2f}")
            self.logger.record("eval/mean_ep_col_penalty", mean_col_penalty)

            mean_num_collisions = None
            if len(self._ep_num_collisions_buffer) > 0:
                mean_num_collisions = np.mean(self._ep_num_collisions_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode number of collisions per episode: {mean_num_collisions:.2f}")
            self.logger.record("eval/mean_ep_num_collisions", mean_num_collisions)

            # Dump log so the evaluation results are printed with the correct timestep
            if self.verbose > 0:
                self.custom_logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"mean episode reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # get the min and max episode reward too
                self.custom_logger.info(f"Min episode reward: {np.min(episode_rewards)}, " f"Max episode reward: {np.max(episode_rewards)}")
                self.custom_logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                self.custom_logger.info(f"Min episode length: {np.min(episode_lengths)}, " f"Max episode length: {np.max(episode_lengths)}")
            # Add to current Logger
            self.logger.record("eval/mean_ep_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)           
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if subgoal_success_rate >= self.best_subgoal_success_rate:
                if self.verbose > 0:
                    self.custom_logger.info("New best success rate!")
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        self.custom_logger.info("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()
                self.best_subgoal_success_rate = subgoal_success_rate
                self.best_mean_reward = mean_reward
            
            if self.recent_model_save_path is not None:
                self.model.save(os.path.join(self.recent_model_save_path, "recent_model"))
            # Save the results in a csv file located in the second to last directory of log_path
            # Split the log_path to get the second to last directory
            csv_path = os.path.split(self.log_path)[0]
            with open(os.path.join(csv_path, 'results_eval.csv'), 'a') as f:
                f.write("{},{},{},{},{},{},{},{}\n".format(
                    self.num_timesteps,  
                    subgoal_success_rate,
                    success_rate, 
                    mean_ep_length,
                    mean_reward,  
                    mean_ep_r_shaping, 
                    mean_num_collisions,
                    mean_col_penalty,  
                ))
                f.close()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
    
    def _log_subgoal_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the subgoal success rate during evaluation.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if locals_["done"] or locals_['truncated']:
            # store the episode reward shaping values and collision penalties
            self._ep_r_shaping_buffer.append(info.get('ep_cumu_r_shaping'))
            self._ep_col_penalty_buffer.append(info.get('ep_cumu_col_penalty'))
            self._ep_num_collisions_buffer.append(info.get('ep_cumu_collisions'))
            maybe_is_success = info.get("goal_success")
            subgoal_success = info.get("subgoal_success")
            #subgoals = [key for key in info.keys() if '_subgoal' in key]
            self._subgoal_successes_buffer.append(subgoal_success)
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)