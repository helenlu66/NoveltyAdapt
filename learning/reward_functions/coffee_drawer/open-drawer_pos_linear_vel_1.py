# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which
            the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which
            the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum possible end‐effector displacement per step, as given
    max_disp = 0.016

    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # progress is reduction in distance between gripper and drawer handle
        prev_vec = prev_observation_with_semantics['drawer1_handle_to_gripper1_pos']
        curr_vec = current_observation_with_semantics['drawer1_handle_to_gripper1_pos']
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # positive reward when gripper moves closer to handle
        raw_reward = (prev_dist - curr_dist) / max_disp
        return float(np.clip(raw_reward, -1.0, 1.0))

    elif grounded_effect == "(open drawer1)":
        # progress is increase in distance between gripper and cabinet (i.e., drawer opening)
        prev_vec = prev_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
        curr_vec = current_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # positive reward when gripper (and thus drawer) moves away from cabinet
        raw_reward = (curr_dist - prev_dist) / max_disp
        return float(np.clip(raw_reward, -1.0, 1.0))

    # if effect unknown or not applicable, no shaping reward
    return 0.0
