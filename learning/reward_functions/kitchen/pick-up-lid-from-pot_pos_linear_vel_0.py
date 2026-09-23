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
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end-effector displacement per step
    max_disp = 0.016

    # helper to clamp between -1 and 1
    def clamp(x):
        return float(np.clip(x, -1.0, 1.0))

    # extract vectors
    prev_pot = np.array(prev_observation_with_semantics["pot1_to_gripper1_pos"])
    cur_pot  = np.array(current_observation_with_semantics["pot1_to_gripper1_pos"])
    prev_lid = np.array(prev_observation_with_semantics["lid1_to_gripper1_pos"])
    cur_lid  = np.array(current_observation_with_semantics["lid1_to_gripper1_pos"])
    prev_handle = np.array(prev_observation_with_semantics["lid1_handle_to_gripper1_pos"])
    cur_handle  = np.array(current_observation_with_semantics["lid1_handle_to_gripper1_pos"])

    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # we want the gripper to close around the lid handle: distance to handle should decrease
        prev_dist = np.linalg.norm(prev_handle)
        cur_dist  = np.linalg.norm(cur_handle)
        # positive reward if we move closer
        return clamp((prev_dist - cur_dist) / max_disp)

    elif grounded_effect == "(not (ontop lid1 pot1))":
        # we want to slide the lid off the pot top: horizontal displacement between lid and pot should increase
        prev_rel = prev_lid - prev_pot
        cur_rel  = cur_lid  - cur_pot
        prev_xy  = np.linalg.norm(prev_rel[:2])
        cur_xy   = np.linalg.norm(cur_rel[:2])
        # positive reward if the lid moves away in the xy‐plane
        return clamp((cur_xy - prev_xy) / max_disp)

    elif grounded_effect == "(not (covered pot1))":
        # we want to lift the lid vertically off the pot: z‐separation should increase
        prev_dz = prev_lid[2] - prev_pot[2]
        cur_dz  = cur_lid[2]  - cur_pot[2]
        # positive reward if the vertical gap increases
        return clamp((cur_dz - prev_dz) / max_disp)

    else:
        # unknown effect: no shaping reward
        return 0.0
