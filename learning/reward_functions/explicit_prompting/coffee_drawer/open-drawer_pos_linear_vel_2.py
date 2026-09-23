# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
            keys have semantics and the values are numpy arrays of numeric values
        current_observation_with_semantics: the current observation after taking the action
            keys have semantics and the values are numpy arrays of numeric values
        grounded_effect (str): the grounded effect of the operator whose progress we are
            trying to measure (one of "(exclusively-occupying-gripper drawer1 gripper1)"
            or "(open drawer1)")
    Returns:
        float: the linear reward between -1 and 1 based on the change toward the grounded effect
    '''
    # maximum possible end-effector displacement per action
    max_disp = 0.016

    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # Progress toward grasping the drawer handle:
        # we measure how the max-coordinate discrepancy (L-infinity norm) between
        # gripper and handle shrinks
        prev_vec = prev_observation_with_semantics['drawer1_handle_to_gripper1_pos']
        curr_vec = current_observation_with_semantics['drawer1_handle_to_gripper1_pos']
        prev_inf = np.max(np.abs(prev_vec))
        curr_inf = np.max(np.abs(curr_vec))
        # positive if we get closer (L-inf decreases), negative if we move away
        progress = (prev_inf - curr_inf) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    elif grounded_effect == "(open drawer1)":
        # Progress toward opening the drawer:
        # we measure how the overall distance (L2 norm) from cabinet to gripper
        # increases as the drawer is pulled out
        prev_vec = prev_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
        curr_vec = current_observation_with_semantics['drawer1_cabinet_to_gripper1_pos']
        prev_dist = np.linalg.norm(prev_vec)
        curr_dist = np.linalg.norm(curr_vec)
        # positive if the gripper (and thus the drawer) moves away from the cabinet
        progress = (curr_dist - prev_dist) / max_disp
        return float(np.clip(progress, -1.0, 1.0))

    else:
        # unrelated effect: no shaping
        return 0.0
