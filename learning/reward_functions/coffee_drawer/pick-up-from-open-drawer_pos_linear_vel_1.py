# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str
) -> float:
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
    # maximum end‐effector displacement per step
    max_disp = 0.016

    # semantic keys we expect in the observations
    # coffee-pod1_to_gripper1_pos : np.ndarray of shape (3,)
    # drawer1_cabinet_to_gripper1_pos : np.ndarray of shape (3,)
    # drawer1_handle_to_gripper1_pos : np.ndarray of shape (3,)
    # gripper1_pos etc. (unused here)

    # shorthand
    prev_obs = prev_observation_with_semantics
    cur_obs = current_observation_with_semantics

    # Effect 1: (exclusively-occupying-gripper coffee-pod1 gripper1)
    #   progress ~ reducing distance between coffee-pod and gripper
    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        prev_dist = np.linalg.norm(prev_obs["coffee-pod1_to_gripper1_pos"])
        cur_dist = np.linalg.norm(cur_obs["coffee-pod1_to_gripper1_pos"])
        # positive reward when coffee-pod moves closer to gripper
        reward = (prev_dist - cur_dist) / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    # Effect 2: (not (in coffee-pod1 drawer1))
    #   progress ~ increasing distance from coffee-pod to drawer cabinet
    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        prev_vec = prev_obs["coffee-pod1_to_gripper1_pos"] - prev_obs["drawer1_cabinet_to_gripper1_pos"]
        cur_vec  = cur_obs["coffee-pod1_to_gripper1_pos"]  - cur_obs["drawer1_cabinet_to_gripper1_pos"]
        prev_dist = np.linalg.norm(prev_vec)
        cur_dist  = np.linalg.norm(cur_vec)
        # positive reward when coffee-pod moves further out of the drawer
        reward = (cur_dist - prev_dist) / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    # any other grounded effect not shaped here
    return 0.0
