# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action 
            in which the keys have semantics and the values are arrays of numeric values
        current_observation_with_semantics: the observation after taking the action
        grounded_effect (str): the grounded effect whose progress we are measuring
    Returns:
        float: a shaped reward between -1 and 1 for the given effect
    '''
    # helper to compute L2 norm
    def l2(x: np.ndarray) -> float:
        return np.linalg.norm(x)

    # maximum end-effector displacement per action
    max_disp = 0.016

    # unpack prev and curr
    prev = prev_observation_with_semantics
    curr = current_observation_with_semantics

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress is reduction in distance from gripper to coffee-pod
        prev_dist = l2(prev["coffee-pod1_to_gripper1_pos"])
        curr_dist = l2(curr["coffee-pod1_to_gripper1_pos"])
        # positive reward if we get closer, normalized by max_disp
        raw = (prev_dist - curr_dist) / max_disp
        return float(np.clip(raw, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # approximate "coffee-pod out of drawer" by moving it away from the drawer handle
        prev_vec = prev["coffee-pod1_to_gripper1_pos"] - prev["drawer1_handle_to_gripper1_pos"]
        curr_vec = curr["coffee-pod1_to_gripper1_pos"] - curr["drawer1_handle_to_gripper1_pos"]
        prev_dist = l2(prev_vec)
        curr_dist = l2(curr_vec)
        # positive reward if pod moves away from the drawer handle
        raw = (curr_dist - prev_dist) / max_disp
        return float(np.clip(raw, -1.0, 1.0))

    else:
        # no shaping for other effects
        return 0.0
