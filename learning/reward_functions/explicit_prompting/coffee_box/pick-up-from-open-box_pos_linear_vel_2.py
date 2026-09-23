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
    # maximum possible end-effector displacement per action
    max_move = 0.016

    # helper to clamp a value between -1 and 1
    def clamp(x):
        return float(np.clip(x, -1.0, 1.0))

    # unpack vectors
    prev_box_to_g = np.array(prev_observation_with_semantics['box1_to_gripper1_pos'])
    curr_box_to_g = np.array(current_observation_with_semantics['box1_to_gripper1_pos'])
    prev_pod_to_g = np.array(prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
    curr_pod_to_g = np.array(current_observation_with_semantics['coffee-pod1_to_gripper1_pos'])

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress = reduction in distance between gripper and pod
        prev_dist = np.linalg.norm(prev_pod_to_g)
        curr_dist = np.linalg.norm(curr_pod_to_g)
        delta = prev_dist - curr_dist
        # normalize by max possible per-step movement
        return clamp(delta / max_move)

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # progress = increase in distance between pod and box
        prev_rel = prev_pod_to_g - prev_box_to_g
        curr_rel = curr_pod_to_g - curr_box_to_g
        prev_dist = np.linalg.norm(prev_rel)
        curr_dist = np.linalg.norm(curr_rel)
        delta = curr_dist - prev_dist
        # normalize by max possible per-step movement
        return clamp(delta / max_move)

    # if effect not recognized, no shaping reward
    return 0.0
