# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end‐effector movement per step
    max_disp = 0.016

    # helper to clip reward
    def clip(r):
        return float(np.clip(r, -1.0, 1.0))

    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # Encourage the lid handle to move into the gripper
        prev_pos = np.array(prev_observation_with_semantics["lid1_handle_to_gripper1_pos"])
        curr_pos = np.array(current_observation_with_semantics["lid1_handle_to_gripper1_pos"])
        prev_dist = np.linalg.norm(prev_pos)
        curr_dist = np.linalg.norm(curr_pos)
        # progress is decrease in distance to gripper
        reward = (prev_dist - curr_dist) / max_disp
        return clip(reward)

    elif grounded_effect == "(not (ontop lid1 pot1))":
        # Encourage lid to move away from pot (i.e., break the "on top" relation)
        prev_lid = np.array(prev_observation_with_semantics["lid1_to_gripper1_pos"])
        prev_pot = np.array(prev_observation_with_semantics["pot1_to_gripper1_pos"])
        curr_lid = np.array(current_observation_with_semantics["lid1_to_gripper1_pos"])
        curr_pot = np.array(current_observation_with_semantics["pot1_to_gripper1_pos"])
        prev_dist = np.linalg.norm(prev_lid - prev_pot)
        curr_dist = np.linalg.norm(curr_lid - curr_pot)
        # progress is increase in lid‐pot distance
        reward = (curr_dist - prev_dist) / max_disp
        return clip(reward)

    elif grounded_effect == "(not (covered pot1))":
        # Also encourage lid to move off the pot (i.e., uncover the pot)
        prev_lid = np.array(prev_observation_with_semantics["lid1_to_gripper1_pos"])
        prev_pot = np.array(prev_observation_with_semantics["pot1_to_gripper1_pos"])
        curr_lid = np.array(current_observation_with_semantics["lid1_to_gripper1_pos"])
        curr_pot = np.array(current_observation_with_semantics["pot1_to_gripper1_pos"])
        prev_dist = np.linalg.norm(prev_lid - prev_pot)
        curr_dist = np.linalg.norm(curr_lid - curr_pot)
        # progress is increase in lid‐pot distance
        reward = (curr_dist - prev_dist) / max_disp
        return clip(reward)

    # If the effect is unrecognized, no shaping
    return 0.0
