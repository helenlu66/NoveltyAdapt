# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics: Dict[str, Union[float, np.ndarray]],
                      current_observation_with_semantics: Dict[str, Union[float, np.ndarray]],
                      grounded_effect: str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
                                        with semantic keys and numpy array values
        current_observation_with_semantics: the current observation after taking the action
                                            with semantic keys and numpy array values
        grounded_effect (str): the grounded effect whose progress we are measuring
    Returns:
        float: the linear reward between -1 and 1 based on progress toward the grounded effect
    '''
    # maximum possible end-effector displacement per step
    max_disp = 0.016

    # extract pod-to-gripper vectors
    prev_pod_vec = prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
    curr_pod_vec = current_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
    prev_dist_pod_gripper = np.linalg.norm(prev_pod_vec)
    curr_dist_pod_gripper = np.linalg.norm(curr_pod_vec)

    # extract box-to-gripper vectors
    prev_box_vec = prev_observation_with_semantics["box1_to_gripper1_pos"]
    curr_box_vec = current_observation_with_semantics["box1_to_gripper1_pos"]

    # compute pod-to-box distances via (pod_to_gripper - box_to_gripper)
    prev_pod_box_vec = prev_pod_vec - prev_box_vec
    curr_pod_box_vec = curr_pod_vec - curr_box_vec
    prev_dist_pod_box = np.linalg.norm(prev_pod_box_vec)
    curr_dist_pod_box = np.linalg.norm(curr_pod_box_vec)

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # reward positive when pod moves closer to gripper
        delta = prev_dist_pod_gripper - curr_dist_pod_gripper
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # reward positive when pod moves away from box
        delta = curr_dist_pod_box - prev_dist_pod_box
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # no shaping for other effects
        return 0.0
