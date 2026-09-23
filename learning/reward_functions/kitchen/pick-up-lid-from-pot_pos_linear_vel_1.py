# llm generated reward shaping function
from typing import Dict, Union
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
    prev_obs = prev_observation_with_semantics
    curr_obs = current_observation_with_semantics
    # maximum possible end‐effector displacement per action
    max_disp = 0.016

    # 1) effect: the lid is grasped by the gripper
    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # distance from lid handle to gripper
        prev_dist = np.linalg.norm(prev_obs["lid1_handle_to_gripper1_pos"])
        curr_dist = np.linalg.norm(curr_obs["lid1_handle_to_gripper1_pos"])
        # positive reward for reducing this distance
        progress = (prev_dist - curr_dist) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    # 2) effect: the lid is no longer on top of the pot
    elif grounded_effect == "(not (ontop lid1 pot1))":
        # compute how far apart lid and pot are, via their distances to the gripper
        prev_diff = abs(
            np.linalg.norm(prev_obs["lid1_to_gripper1_pos"])
            - np.linalg.norm(prev_obs["pot1_to_gripper1_pos"])
        )
        curr_diff = abs(
            np.linalg.norm(curr_obs["lid1_to_gripper1_pos"])
            - np.linalg.norm(curr_obs["pot1_to_gripper1_pos"])
        )
        # positive reward for increasing this separation
        progress = (curr_diff - prev_diff) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    # 3) effect: the pot is uncovered (i.e., lid moved away from pot)
    elif grounded_effect == "(not (covered pot1))":
        # similar metric to lifting lid off pot
        prev_diff = abs(
            np.linalg.norm(prev_obs["lid1_to_gripper1_pos"])
            - np.linalg.norm(prev_obs["pot1_to_gripper1_pos"])
        )
        curr_diff = abs(
            np.linalg.norm(curr_obs["lid1_to_gripper1_pos"])
            - np.linalg.norm(curr_obs["pot1_to_gripper1_pos"])
        )
        # reward proportional to further uncovering
        progress = (curr_diff - prev_diff) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    # if the effect is unrecognized, no shaping reward
    else:
        return 0.0
