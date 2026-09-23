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
        current_observation_with_semantics: the current observation after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum possible displacement of any object relative to gripper per step
    max_disp = 0.016

    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # progress is how much closer the gripper handle gets to the square nut's handle
        prev_dist = np.linalg.norm(
            prev_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        )
        curr_dist = np.linalg.norm(
            current_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        )
        # positive reward if distance decreases
        reward = (prev_dist - curr_dist) / max_disp
        # clip to [-1, 1]
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # progress is how much further the square nut moves away from the round peg
        # reconstruct world positions of square nut and round peg from gripper-relative vectors
        prev_gripper = prev_observation_with_semantics["gripper1_pos"]
        curr_gripper = current_observation_with_semantics["gripper1_pos"]

        prev_sq = prev_gripper - prev_observation_with_semantics["square-nut1_to_gripper1_pos"]
        prev_pg = prev_gripper - prev_observation_with_semantics["round-peg1_to_gripper1_pos"]
        curr_sq = curr_gripper - current_observation_with_semantics["square-nut1_to_gripper1_pos"]
        curr_pg = curr_gripper - current_observation_with_semantics["round-peg1_to_gripper1_pos"]

        prev_dist = np.linalg.norm(prev_sq - prev_pg)
        curr_dist = np.linalg.norm(curr_sq - curr_pg)
        # positive reward if peg-nut distance increases
        reward = (curr_dist - prev_dist) / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # no shaping for other effects
        return 0.0
