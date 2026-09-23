# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action 
            keys: 'gripper1_pos', 'pot1_to_gripper1_pos', 'lid1_to_gripper1_pos', 'lid1_handle_to_gripper1_pos'
        current_observation_with_semantics: the current observation after the action
        grounded_effect (str): the grounded effect of the operator whose progress we are tracking
    Returns:
        float: a shaped reward in [-1, 1] measuring progress toward the grounded effect
    '''
    # maximum possible change in any relative distance per step
    MAX_DISP = 0.016
    
    # helper to clip to [-1,1]
    def clip(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    # fetch the relevant vectors
    prev_lid_handle = np.asarray(prev_observation_with_semantics['lid1_handle_to_gripper1_pos'])
    curr_lid_handle = np.asarray(current_observation_with_semantics['lid1_handle_to_gripper1_pos'])
    prev_lid = np.asarray(prev_observation_with_semantics['lid1_to_gripper1_pos'])
    curr_lid = np.asarray(current_observation_with_semantics['lid1_to_gripper1_pos'])
    prev_pot = np.asarray(prev_observation_with_semantics['pot1_to_gripper1_pos'])
    curr_pot = np.asarray(current_observation_with_semantics['pot1_to_gripper1_pos'])

    # 1) Exclusively occupy the gripper with the lid:
    #    minimize the distance from the gripper to the lid's handle
    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        d_prev = np.linalg.norm(prev_lid_handle)
        d_curr = np.linalg.norm(curr_lid_handle)
        # positive reward if we moved the gripper closer to the lid handle
        return clip((d_prev - d_curr) / MAX_DISP)

    # 2) Remove the lid from on top of the pot:
    #    maximize the lid-to-pot distance
    elif grounded_effect == "(not (ontop lid1 pot1))":
        d_prev = np.linalg.norm(prev_lid - prev_pot)
        d_curr = np.linalg.norm(curr_lid - curr_pot)
        # positive reward if lid-pot distance increased
        return clip((d_curr - d_prev) / MAX_DISP)

    # 3) Uncover the pot (i.e., lid no longer covers the pot):
    #    same metric as lifting lid off the pot
    elif grounded_effect == "(not (covered pot1))":
        d_prev = np.linalg.norm(prev_lid - prev_pot)
        d_curr = np.linalg.norm(curr_lid - curr_pot)
        # positive reward if lid-pot distance increased
        return clip((d_curr - d_prev) / MAX_DISP)

    # if effect is unknown, no shaping reward
    return 0.0
