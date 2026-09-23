# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys include:
              - "gripper1_pos": np.array(3,)
              - "drawer1_cabinet_to_gripper1_pos": np.array(3,)
              - "drawer1_handle_to_gripper1_pos": np.array(3,)
              - "coffee-pod1_to_gripper1_pos": np.array(3,)
        current_observation_with_semantics: the current observation after the action
            same keys as above
        grounded_effect (str): the grounded effect of the operator whose progress we measure
    Returns:
        float: a dense shaping reward in [-1, 1] for progress toward the effect
    '''
    # maximum end‐effector displacement per action
    max_disp = 0.016

    # helper to compute Euclidean norm
    def norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    # previous and current distance: coffee-pod1 <-> gripper1
    prev_coffee_gripper_dist = norm(prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
    curr_coffee_gripper_dist = norm(current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])

    # previous and current distance: coffee-pod1 <-> drawer1 cabinet
    # = |(coffee->gripper) - (drawer->gripper)|
    prev_coffee_drawer_dist = norm(
        prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        - prev_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"]
    )
    curr_coffee_drawer_dist = norm(
        current_observation_with_semantics["coffee-pod1_to_gripper1_pos"]
        - current_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"]
    )

    # shaping for each grounded effect
    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # reward progress when coffee-pod1 moves closer to the gripper
        delta = (prev_coffee_gripper_dist - curr_coffee_gripper_dist) / max_disp
        return float(np.clip(delta, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # reward progress when coffee-pod1 moves away from the drawer cabinet
        delta = (curr_coffee_drawer_dist - prev_coffee_drawer_dist) / max_disp
        return float(np.clip(delta, -1.0, 1.0))

    # if the effect is unrecognized, no shaping
    return 0.0
