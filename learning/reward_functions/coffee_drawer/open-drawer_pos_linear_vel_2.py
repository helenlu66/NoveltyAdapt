# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action
        current_observation_with_semantics: the current observation after taking the action
        grounded_effect: the grounded effect whose progress we measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end‐effector displacement per step (given)
    max_disp = 0.016

    # distance from gripper to drawer handle
    prev_hdl_dist = np.linalg.norm(prev_observation_with_semantics['drawer1_handle_to_gripper1_pos'])
    curr_hdl_dist = np.linalg.norm(current_observation_with_semantics['drawer1_handle_to_gripper1_pos'])

    # distance between drawer handle and cabinet faces (i.e., how far drawer is pulled out)
    prev_handle_cabinet_vec = (prev_observation_with_semantics['drawer1_handle_to_gripper1_pos'] -
                               prev_observation_with_semantics['drawer1_cabinet_to_gripper1_pos'])
    curr_handle_cabinet_vec = (current_observation_with_semantics['drawer1_handle_to_gripper1_pos'] -
                               current_observation_with_semantics['drawer1_cabinet_to_gripper1_pos'])
    prev_handle_cabinet_dist = np.linalg.norm(prev_handle_cabinet_vec)
    curr_handle_cabinet_dist = np.linalg.norm(curr_handle_cabinet_vec)

    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # we want the gripper to move closer to the handle: reduction in handle distance
        delta = prev_hdl_dist - curr_hdl_dist
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(open drawer1)":
        # we want the drawer to open: increase in handle-to-cabinet distance
        delta = curr_handle_cabinet_dist - prev_handle_cabinet_dist
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # unknown effect: no shaping reward
        return 0.0
