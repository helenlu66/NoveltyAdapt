# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics 
                                        and the values are arrays of numeric values
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics 
                                           and the values are arrays of numeric values
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end-effector movement per action
    max_disp = 0.016

    def l2(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    # distance between two objects A and B based on their reported vectors to the gripper
    def inter_object_distance(keyA: str, keyB: str, obs: Dict[str, np.ndarray]) -> float:
        return abs(l2(obs[keyA]) - l2(obs[keyB]))

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # progress = reduction in distance between gripper and coffee-pod1
        prev_dist = l2(prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
        curr_dist = l2(current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
        delta = (prev_dist - curr_dist) / max_disp
        return float(np.clip(delta, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # progress = coffee-pod1 moving away from drawer1 (measured via difference in distances to gripper)
        prev_sep = inter_object_distance("coffee-pod1_to_gripper1_pos",
                                         "drawer1_handle_to_gripper1_pos",
                                         prev_observation_with_semantics)
        curr_sep = inter_object_distance("coffee-pod1_to_gripper1_pos",
                                         "drawer1_handle_to_gripper1_pos",
                                         current_observation_with_semantics)
        delta = (curr_sep - prev_sep) / max_disp
        return float(np.clip(delta, -1.0, 1.0))

    else:
        # unknown effect: no shaping
        return 0.0
