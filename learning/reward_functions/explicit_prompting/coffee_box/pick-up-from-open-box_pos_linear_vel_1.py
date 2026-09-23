# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys
            have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys
            have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum possible gripper displacement per action (given)
    max_disp = 0.016

    # helper to compute Euclidean norm
    def norm(x: np.ndarray) -> float:
        return float(np.linalg.norm(x))

    # compute distances in previous and current observations
    prev_coffee_gripper = norm(prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
    curr_coffee_gripper = norm(current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
    prev_box_gripper = norm(prev_observation_with_semantics["box1_to_gripper1_pos"])
    curr_box_gripper = norm(current_observation_with_semantics["box1_to_gripper1_pos"])

    # infer coffee-pod to box1 distance via difference of relative-to-gripper vectors
    prev_box_pod = norm(prev_observation_with_semantics["box1_to_gripper1_pos"] -
                        prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
    curr_box_pod = norm(current_observation_with_semantics["box1_to_gripper1_pos"] -
                        current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])

    reward = 0.0

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress in closing the distance to the coffee-pod
        # positive when we get closer, negative when we move away
        delta = prev_coffee_gripper - curr_coffee_gripper
        reward = np.clip(delta / max_disp, -1.0, 1.0)

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # progress in pulling the pod out of the box
        # coffee-pod exits the box => increase in box-to-pod distance
        delta = curr_box_pod - prev_box_pod
        reward = np.clip(delta / max_disp, -1.0, 1.0)

    else:
        # unknown effect: zero shaping
        reward = 0.0

    return reward
