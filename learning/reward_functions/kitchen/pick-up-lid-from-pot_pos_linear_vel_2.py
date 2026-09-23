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
    # maximum end‐effector displacement per action
    max_disp = 0.016

    # helper to clip to [-1,1]
    def clip01(x: float) -> float:
        return float(np.maximum(-1.0, np.minimum(1.0, x)))

    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # progress = reduction in distance between lid handle and gripper
        prev_d = np.linalg.norm(prev_observation_with_semantics['lid1_handle_to_gripper1_pos'])
        curr_d = np.linalg.norm(current_observation_with_semantics['lid1_handle_to_gripper1_pos'])
        delta = prev_d - curr_d
        return clip01(delta / max_disp)

    elif grounded_effect == "(not (ontop lid1 pot1))":
        # progress = increase in separation between lid and pot
        prev_vec = (prev_observation_with_semantics['lid1_to_gripper1_pos'] -
                    prev_observation_with_semantics['pot1_to_gripper1_pos'])
        curr_vec = (current_observation_with_semantics['lid1_to_gripper1_pos'] -
                    current_observation_with_semantics['pot1_to_gripper1_pos'])
        prev_d = np.linalg.norm(prev_vec)
        curr_d = np.linalg.norm(curr_vec)
        delta = curr_d - prev_d
        return clip01(delta / max_disp)

    elif grounded_effect == "(not (covered pot1))":
        # same geometric cue as lifting lid off pot
        prev_vec = (prev_observation_with_semantics['lid1_to_gripper1_pos'] -
                    prev_observation_with_semantics['pot1_to_gripper1_pos'])
        curr_vec = (current_observation_with_semantics['lid1_to_gripper1_pos'] -
                    current_observation_with_semantics['pot1_to_gripper1_pos'])
        prev_d = np.linalg.norm(prev_vec)
        curr_d = np.linalg.norm(curr_vec)
        delta = curr_d - prev_d
        return clip01(delta / max_disp)

    # default: no shaping for other effects
    return 0.0
