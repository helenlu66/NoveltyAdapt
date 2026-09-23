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
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum possible end-effector displacement per action
    max_disp = 0.016

    # 1) Effect: gripper exclusively occupies the coffee pod → distance gripper↔pod should decrease
    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        prev_dist = np.linalg.norm(prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
        curr_dist = np.linalg.norm(current_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
        # positive reward for pulling the pod closer
        progress = (prev_dist - curr_dist) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    # 2) Effect: coffee pod not in box → distance pod↔box1 should increase
    elif grounded_effect == "(not (in coffee-pod1 box1))":
        prev_vec = (prev_observation_with_semantics['coffee-pod1_to_gripper1_pos']
                    - prev_observation_with_semantics['box1_to_gripper1_pos'])
        curr_vec = (current_observation_with_semantics['coffee-pod1_to_gripper1_pos']
                    - current_observation_with_semantics['box1_to_gripper1_pos'])
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # positive reward for moving the pod away from the box
        progress = (curr_dist - prev_dist) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    # If the effect is unrecognized, no shaping reward
    return 0.0
