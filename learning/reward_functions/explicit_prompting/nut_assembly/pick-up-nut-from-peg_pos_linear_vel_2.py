# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum possible displacement of the end effector per action
    max_disp = 0.016

    # helper to clip reward to [-1, 1]
    def clip(x):
        return float(np.clip(x, -1.0, 1.0))

    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # measure how much closer the nut handle moved to the gripper
        prev_handle_vec = prev_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        curr_handle_vec = current_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        prev_dist = np.linalg.norm(prev_handle_vec)
        curr_dist = np.linalg.norm(curr_handle_vec)
        # positive reward when distance decreases
        reward = (prev_dist - curr_dist) / max_disp
        return clip(reward)

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # measure how much farther the nut moved from the peg
        prev_nut_vec = prev_observation_with_semantics["square-nut1_to_gripper1_pos"]
        prev_peg_vec = prev_observation_with_semantics["round-peg1_to_gripper1_pos"]
        curr_nut_vec = current_observation_with_semantics["square-nut1_to_gripper1_pos"]
        curr_peg_vec = current_observation_with_semantics["round-peg1_to_gripper1_pos"]
        prev_rel = prev_nut_vec - prev_peg_vec
        curr_rel = curr_nut_vec - curr_peg_vec
        prev_dist = np.linalg.norm(prev_rel)
        curr_dist = np.linalg.norm(curr_rel)
        # positive reward when distance increases
        reward = (curr_dist - prev_dist) / max_disp
        return clip(reward)

    else:
        # no shaping for other effects
        return 0.0
