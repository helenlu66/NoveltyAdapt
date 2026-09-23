# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics: Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics: Dict[str, Union[bool, float, np.array]],
                      grounded_effect: str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys: 'gripper1_pos', 'box1_to_gripper1_pos', 'coffee-pod1_to_gripper1_pos'
        current_observation_with_semantics: the current observation after taking the action
            same keys as above
        grounded_effect (str): the grounded effect of the operator
            either "(exclusively-occupying-gripper coffee-pod1 gripper1)"
            or "(not (in coffee-pod1 box1))"
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # helper to clamp reward to [-1,1]
    def clamp(x: float) -> float:
        return max(-1.0, min(1.0, x))

    # extract vectors
    prev_pod_gripper = np.linalg.norm(
        prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
    cur_pod_gripper = np.linalg.norm(
        current_observation_with_semantics['coffee-pod1_to_gripper1_pos'])

    # relative pod-to-box distance via their gripper-relative positions
    prev_pod_box = np.linalg.norm(
        prev_observation_with_semantics['coffee-pod1_to_gripper1_pos']
        - prev_observation_with_semantics['box1_to_gripper1_pos']
    )
    cur_pod_box = np.linalg.norm(
        current_observation_with_semantics['coffee-pod1_to_gripper1_pos']
        - current_observation_with_semantics['box1_to_gripper1_pos']
    )

    # maximum end-effector displacement per step
    max_disp = 0.016

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress = decrease in pod->gripper distance (we want pod closer to gripper)
        delta = (prev_pod_gripper - cur_pod_gripper) / max_disp
        return clamp(delta)

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # progress = increase in pod->box distance (we want pod further from box)
        delta = (cur_pod_box - prev_pod_box) / max_disp
        return clamp(delta)

    else:
        # unknown effect: no shaping reward
        return 0.0
