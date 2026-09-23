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
        current_observation_with_semantics: the current observation after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we measure
    Returns:
        float: the linear reward between -1 and 1 based on progress toward the grounded effect
    '''
    # maximal end-effector displacement per step (given)
    max_disp = 0.016
    
    # helper to clip reward
    def clip(r): 
        return float(max(-1.0, min(1.0, r)))
    
    # 1) Grasping the square nut: distance from gripper to nut handle should decrease
    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        prev_vec = prev_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        cur_vec  = current_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        prev_dist = np.linalg.norm(prev_vec)
        cur_dist  = np.linalg.norm(cur_vec)
        # positive reward if we close in, negative if we drift away
        reward = (prev_dist - cur_dist) / max_disp
        return clip(reward)
    
    # 2) Removing the nut from the peg: distance nut↔peg should increase
    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        prev_n2g = prev_observation_with_semantics["square-nut1_to_gripper1_pos"]
        prev_p2g = prev_observation_with_semantics["round-peg1_to_gripper1_pos"]
        cur_n2g  = current_observation_with_semantics["square-nut1_to_gripper1_pos"]
        cur_p2g  = current_observation_with_semantics["round-peg1_to_gripper1_pos"]
        prev_vec = prev_n2g - prev_p2g
        cur_vec  = cur_n2g  - cur_p2g
        prev_dist = np.linalg.norm(prev_vec)
        cur_dist  = np.linalg.norm(cur_vec)
        # positive reward if nut moves away from peg, negative if it moves closer
        reward = (cur_dist - prev_dist) / max_disp
        return clip(reward)
    
    # if the effect is unrecognized, no shaping
    return 0.0
