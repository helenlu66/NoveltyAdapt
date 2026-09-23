# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.array]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.array]],
    grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
        current_observation_with_semantics: the current observation after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end‐effector displacement per step
    max_disp = 0.016

    # helper to clip reward
    def clip(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    # unpack previous and current observations
    prev = prev_observation_with_semantics
    curr = current_observation_with_semantics

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # reward progress as reduction in distance between coffee-pod1 and gripper1
        prev_vec = prev["coffee-pod1_to_gripper1_pos"]
        curr_vec = curr["coffee-pod1_to_gripper1_pos"]
        d_prev = np.linalg.norm(prev_vec)
        d_curr = np.linalg.norm(curr_vec)
        # positive reward when d_curr < d_prev
        reward = (d_prev - d_curr) / max_disp
        return clip(reward)

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # reward progress as increase in distance between coffee-pod1 and drawer1 cabinet
        prev_diff = prev["coffee-pod1_to_gripper1_pos"] - prev["drawer1_cabinet_to_gripper1_pos"]
        curr_diff = curr["coffee-pod1_to_gripper1_pos"] - curr["drawer1_cabinet_to_gripper1_pos"]
        d_prev = np.linalg.norm(prev_diff)
        d_curr = np.linalg.norm(curr_diff)
        # positive reward when d_curr > d_prev
        reward = (d_curr - d_prev) / max_disp
        return clip(reward)

    else:
        # no shaping for other effects
        return 0.0
