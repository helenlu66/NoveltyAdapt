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
            keys: 'gripper1_pos', 'drawer1_cabinet_to_gripper1_pos', 'drawer1_handle_to_gripper1_pos'
        current_observation_with_semantics: the current observation after taking the action
            same keys as prev_observation_with_semantics
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end‐effector displacement per step (given)
    max_disp = 0.016

    # compute distance from handle to gripper before and after action
    prev_handle_vec = prev_observation_with_semantics['drawer1_handle_to_gripper1_pos']
    curr_handle_vec = current_observation_with_semantics['drawer1_handle_to_gripper1_pos']
    prev_handle_dist = float(np.linalg.norm(prev_handle_vec))
    curr_handle_dist = float(np.linalg.norm(curr_handle_vec))
    # compute distance from cabinet to gripper before and after action
    prev_cabinet_vec = prev_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
    curr_cabinet_vec = current_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
    prev_cabinet_dist = float(np.linalg.norm(prev_cabinet_vec))
    curr_cabinet_dist = float(np.linalg.norm(curr_cabinet_vec))

    # reward for closing in on the handle (for exclusive occupancy effect)
    # progress = reduction in distance to handle, normalized by max_disp
    handle_progress = (prev_handle_dist - curr_handle_dist) / max_disp
    handle_progress = float(np.clip(handle_progress, -1.0, 1.0))

    # reward for opening the drawer (for open-drawer effect)
    # progress = increase in distance from cabinet to gripper, normalized by max_disp
    open_progress = (curr_cabinet_dist - prev_cabinet_dist) / max_disp
    open_progress = float(np.clip(open_progress, -1.0, 1.0))

    # dispatch based on which grounded effect we are measuring
    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        return handle_progress
    elif grounded_effect == "(open drawer1)":
        return open_progress
    else:
        # unknown effect: no shaping
        return 0.0
