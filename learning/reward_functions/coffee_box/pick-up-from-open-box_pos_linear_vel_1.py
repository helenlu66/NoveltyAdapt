# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
                      current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
                      grounded_effect: str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys: 'gripper1_pos', 'box1_to_gripper1_pos', 'coffee-pod1_to_gripper1_pos'
        current_observation_with_semantics: the current observation after taking the action
            same keys as above
        grounded_effect (str): the grounded effect whose progress we are measuring
    Returns:
        float: dense shaping reward between -1 and 1
    '''
    # max end-effector movement per action (given)
    max_disp = 0.016

    # extract relative position vectors
    prev_box_to_gripper = prev_observation_with_semantics['box1_to_gripper1_pos']
    prev_pod_to_gripper = prev_observation_with_semantics['coffee-pod1_to_gripper1_pos']
    curr_box_to_gripper = current_observation_with_semantics['box1_to_gripper1_pos']
    curr_pod_to_gripper = current_observation_with_semantics['coffee-pod1_to_gripper1_pos']

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # reward progress as coffee-pod gets closer to the gripper
        prev_dist = np.linalg.norm(prev_pod_to_gripper)
        curr_dist = np.linalg.norm(curr_pod_to_gripper)
        delta = prev_dist - curr_dist
        reward = np.clip(delta / max_disp, -1.0, 1.0)
        return float(reward)

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # reward progress as coffee-pod moves away from the box
        prev_box_to_pod = prev_pod_to_gripper - prev_box_to_gripper
        curr_box_to_pod = curr_pod_to_gripper - curr_box_to_gripper
        prev_dist = np.linalg.norm(prev_box_to_pod)
        curr_dist = np.linalg.norm(curr_box_to_pod)
        delta = curr_dist - prev_dist
        reward = np.clip(delta / max_disp, -1.0, 1.0)
        return float(reward)

    else:
        # no shaping for other effects
        return 0.0
