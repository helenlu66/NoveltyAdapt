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
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # Maximum possible end‐effector displacement per action
    max_disp = 0.016

    # helper to compute euclidean norm
    def norm(v: np.array) -> float:
        return float(np.linalg.norm(v))

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress = bringing coffee-pod1 closer to gripper1
        prev_dist = norm(prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
        curr_dist = norm(current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
        # positive reward when distance decreases
        delta = prev_dist - curr_dist
        return float(np.clip(delta / max_disp, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # progress = moving coffee-pod1 out of box1
        # compute coffee-pod1 to box1 distance via their gripper-relative vectors
        prev_rel = (
            prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
            - prev_observation_with_semantics["box1_to_gripper1_pos"]
        )
        curr_rel = (
            current_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
            - current_observation_with_semantics["box1_to_gripper1_pos"]
        )
        prev_dist = norm(prev_rel)
        curr_dist = norm(curr_rel)
        # positive reward when coffee-pod1 to box1 distance increases
        delta = curr_dist - prev_dist
        return float(np.clip(delta / max_disp, -1.0, 1.0))

    else:
        # if some other effect, no shaping
        return 0.0
