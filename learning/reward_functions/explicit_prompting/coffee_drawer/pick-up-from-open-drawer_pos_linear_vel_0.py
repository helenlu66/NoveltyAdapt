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
    # maximum displacement of end effector per action (meters)
    max_disp = 0.016

    # helper to clip reward to [-1, 1]
    def clip(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress: coffee-pod1 should move toward gripper
        prev_vec = prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        curr_vec = current_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # positive if we reduce the distance
        delta = (prev_dist - curr_dist) / max_disp
        return clip(delta)

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # progress: coffee-pod1 should move away from drawer1 cabinet
        p_coffee = prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        c_coffee = current_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        p_drawer = prev_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"]
        c_drawer = current_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"]
        prev_dist = np.linalg.norm(p_drawer - p_coffee)
        curr_dist = np.linalg.norm(c_drawer - c_coffee)
        # positive if we increase the distance
        delta = (curr_dist - prev_dist) / max_disp
        return clip(delta)

    else:
        # no reward shaping for other effects
        return 0.0
