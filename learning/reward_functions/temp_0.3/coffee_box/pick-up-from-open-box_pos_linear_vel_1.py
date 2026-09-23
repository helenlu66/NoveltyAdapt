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
            keys: 'gripper1_pos', 'box1_to_gripper1_pos', 'coffee-pod1_to_gripper1_pos'
        current_observation_with_semantics: the current observation after taking the action
            same keys as prev_observation_with_semantics
        grounded_effect (str): the grounded effect string to shape progress for
            either "(exclusively-occupying-gripper coffee-pod1 gripper1)"
            or "(not (in coffee-pod1 box1))"
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end-effector displacement per action
    max_disp = 0.016

    # extract relevant vectors
    prev_pod_vec = np.array(prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
    curr_pod_vec = np.array(current_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
    prev_box_vec = np.array(prev_observation_with_semantics['box1_to_gripper1_pos'])
    curr_box_vec = np.array(current_observation_with_semantics['box1_to_gripper1_pos'])

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress is reduction in distance between gripper and coffee-pod1
        prev_dist = np.linalg.norm(prev_pod_vec)
        curr_dist = np.linalg.norm(curr_pod_vec)
        diff = prev_dist - curr_dist
        # normalize to [-1,1]
        reward = diff / max_disp
        # clamp
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # progress is increase in distance between coffee-pod1 and box1 centers
        # vector from coffee-pod1 to box1 = box1_to_gripper - coffee-pod1_to_gripper
        prev_sep = prev_box_vec - prev_pod_vec
        curr_sep = curr_box_vec - curr_pod_vec
        prev_dist = np.linalg.norm(prev_sep)
        curr_dist = np.linalg.norm(curr_sep)
        diff = curr_dist - prev_dist
        # normalize to [-1,1]
        reward = diff / max_disp
        # clamp
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # no shaping for other effects
        return 0.0
