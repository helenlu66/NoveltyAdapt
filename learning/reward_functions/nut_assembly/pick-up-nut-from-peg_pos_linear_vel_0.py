# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics  : Dict[str, Union[bool, float, np.ndarray]],
                      current_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
                      grounded_effect                     : str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys include:
              - "gripper1_pos": end‐effector position (unused here)
              - "square-nut1_to_gripper1_pos": vector from gripper to nut
              - "square-nut1_handle_to_gripper1_pos": vector from gripper to nut handle
              - "round-peg1_to_gripper1_pos": vector from gripper to peg
        current_observation_with_semantics: the current observation after the action
        grounded_effect (str): one of
            "(exclusively-occupying-gripper square-nut1 gripper1)"
            "(not (on-peg square-nut1 round-peg1))"
    Returns:
        float: the shaped reward in [-1, 1] indicating progress toward that effect
    '''
    # maximum per-step movement of any object relative to the gripper is 0.016
    # so the maximum change in the distance between two objects via the gripper is 0.032
    max_disp_single = 0.016
    max_disp_pair   = 2 * max_disp_single

    # pull out arrays
    prev = prev_observation_with_semantics
    curr = current_observation_with_semantics

    reward = 0.0

    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # we expect the handle of the nut to move closer to the gripper
        prev_handle_vec = np.array(prev["square-nut1_handle_to_gripper1_pos"])
        curr_handle_vec = np.array(curr["square-nut1_handle_to_gripper1_pos"])
        prev_dist = np.linalg.norm(prev_handle_vec)
        curr_dist = np.linalg.norm(curr_handle_vec)
        # positive if distance decreases
        delta = prev_dist - curr_dist
        reward = delta / max_disp_single

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # we expect the nut to move away from the peg
        prev_nut_vec = np.array(prev["square-nut1_to_gripper1_pos"])
        curr_nut_vec = np.array(curr["square-nut1_to_gripper1_pos"])
        prev_peg_vec = np.array(prev["round-peg1_to_gripper1_pos"])
        curr_peg_vec = np.array(curr["round-peg1_to_gripper1_pos"])
        prev_diff = prev_nut_vec - prev_peg_vec
        curr_diff = curr_nut_vec - curr_peg_vec
        prev_dist = np.linalg.norm(prev_diff)
        curr_dist = np.linalg.norm(curr_diff)
        # positive if nut-peg distance increases
        delta = curr_dist - prev_dist
        reward = delta / max_disp_pair

    # clamp to [-1, 1]
    return float(np.clip(reward, -1.0, 1.0))
