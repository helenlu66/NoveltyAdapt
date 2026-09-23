# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics
            and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics
            and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    prev_obs = prev_observation_with_semantics
    curr_obs = current_observation_with_semantics

    # maximum end‐effector displacement per step
    max_disp = 0.016

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress = how much closer the gripper got to the coffee pod
        prev_d = np.linalg.norm(prev_obs['coffee-pod1_to_gripper1_pos'])
        curr_d = np.linalg.norm(curr_obs['coffee-pod1_to_gripper1_pos'])
        delta = prev_d - curr_d
        # normalize into [-1,1]
        return float(np.clip(delta / max_disp, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # progress = how much the coffee pod moved away from the drawer handle
        # using relative-to-gripper vectors: 
        prev_rel = prev_obs['drawer1_handle_to_gripper1_pos'] - prev_obs['coffee-pod1_to_gripper1_pos']
        curr_rel = curr_obs['drawer1_handle_to_gripper1_pos'] - curr_obs['coffee-pod1_to_gripper1_pos']
        prev_d = np.linalg.norm(prev_rel)
        curr_d = np.linalg.norm(curr_rel)
        delta = curr_d - prev_d
        # normalize into [-1,1]
        return float(np.clip(delta / max_disp, -1.0, 1.0))

    else:
        # unknown effect → no shaping
        return 0.0
