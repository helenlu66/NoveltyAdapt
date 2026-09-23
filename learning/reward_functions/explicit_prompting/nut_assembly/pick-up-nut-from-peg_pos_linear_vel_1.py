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
        prev_observation_with_semantics: the previous observation before taking the action in which
                                         the keys have semantics and the values are arrays of numeric values
        current_observation_with_semantics: the current observation after taking the action in which
                                           the keys have semantics and the values are arrays of numeric values
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on progress toward the grounded effect
    '''
    # We only have the following semantic position keys:
    #   'square-nut1_to_gripper1_pos'
    #   'square-nut1_handle_to_gripper1_pos'
    #   'round-peg1_to_gripper1_pos'
    # Maximum possible end-effector displacement per action step
    MAX_DISPLACEMENT = 0.016

    prev = prev_observation_with_semantics
    curr = current_observation_with_semantics

    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # Measure distance between the square-nut's handle and the gripper
        prev_dist = np.linalg.norm(prev["square-nut1_handle_to_gripper1_pos"])
        curr_dist = np.linalg.norm(curr["square-nut1_handle_to_gripper1_pos"])
        # Positive reward if the nut handle moves closer to the gripper
        reward = (prev_dist - curr_dist) / MAX_DISPLACEMENT
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # Measure distance between square-nut and round-peg via differences to gripper
        prev_vec = prev["square-nut1_to_gripper1_pos"] - prev["round-peg1_to_gripper1_pos"]
        curr_vec = curr["square-nut1_to_gripper1_pos"] - curr["round-peg1_to_gripper1_pos"]
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # Positive reward if the nut moves away from the peg
        reward = (curr_dist - prev_dist) / MAX_DISPLACEMENT
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # No shaping for other effects
        return 0.0
