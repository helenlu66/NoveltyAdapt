# llm generated reward shaping function
from typing import Dict, Union
import numpy as np

def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys have semantics and values are numeric arrays or scalars
        current_observation_with_semantics: the current observation after taking the action
            keys have semantics and values are numeric arrays or scalars
        grounded_effect (str): the grounded effect of the operator whose progress we are measuring
    Returns:
        float: a linear reward in [-1, 1] estimating progress toward the grounded effect
    '''
    # maximum possible EE displacement per action
    max_disp = 0.016

    # helper: clip progress to [-1, 1]
    def clip(x):
        return float(np.clip(x, -1.0, 1.0))

    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # progress = reduction in distance from gripper to drawer handle
        prev_handle_vec = np.asarray(prev_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        curr_handle_vec = np.asarray(current_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        prev_dist = np.linalg.norm(prev_handle_vec)
        curr_dist = np.linalg.norm(curr_handle_vec)
        # positive when gripper moves closer to handle
        delta = (prev_dist - curr_dist) / max_disp
        return clip(delta)

    elif grounded_effect == "(open drawer1)":
        # progress = increase in distance between drawer handle and cabinet
        prev_cab_vec = np.asarray(prev_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"])
        prev_handle_vec = np.asarray(prev_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        curr_cab_vec = np.asarray(current_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"])
        curr_handle_vec = np.asarray(current_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        # cabinet-to-handle vector = cabinet_to_gripper - handle_to_gripper
        prev_c2h = prev_cab_vec - prev_handle_vec
        curr_c2h = curr_cab_vec - curr_handle_vec
        prev_gap = np.linalg.norm(prev_c2h)
        curr_gap = np.linalg.norm(curr_c2h)
        # positive when handle moves outward (drawer opening)
        delta = (curr_gap - prev_gap) / max_disp
        return clip(delta)

    else:
        # unknown effect: no shaping
        return 0.0
