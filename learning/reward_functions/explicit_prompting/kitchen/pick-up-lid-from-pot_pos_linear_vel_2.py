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
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics 
                                        and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics 
                                           and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end-effector travel per action
    max_disp = 0.016

    # helper to safe‐clip
    def clip01(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    # extract vectors from observations
    prev = prev_observation_with_semantics
    curr = current_observation_with_semantics

    # Effect 1: gripper must pick up lid1 => distance from gripper to lid1 handle decreases
    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # handle position relative to gripper
        prev_handle = np.array(prev["lid1_handle_to_gripper1_pos"], dtype=float)
        curr_handle = np.array(curr["lid1_handle_to_gripper1_pos"], dtype=float)
        prev_dist = np.linalg.norm(prev_handle)
        curr_dist = np.linalg.norm(curr_handle)
        # positive reward if we moved closer to the handle
        progress = (prev_dist - curr_dist) / max_disp
        return clip01(progress)

    # Effect 2: lid1 must no longer be on top of pot1 => lid‐pot separation increases
    elif grounded_effect == "(not (ontop lid1 pot1))":
        prev_lid = np.array(prev["lid1_to_gripper1_pos"], dtype=float)
        prev_pot = np.array(prev["pot1_to_gripper1_pos"], dtype=float)
        curr_lid = np.array(curr["lid1_to_gripper1_pos"], dtype=float)
        curr_pot = np.array(curr["pot1_to_gripper1_pos"], dtype=float)
        prev_sep = np.linalg.norm(prev_lid - prev_pot)
        curr_sep = np.linalg.norm(curr_lid - curr_pot)
        # positive reward if lid and pot move further apart
        progress = (curr_sep - prev_sep) / max_disp
        return clip01(progress)

    # Effect 3: pot1 must no longer be covered => same geometric cue as lid‐pot separation
    elif grounded_effect == "(not (covered pot1))":
        prev_lid = np.array(prev["lid1_to_gripper1_pos"], dtype=float)
        prev_pot = np.array(prev["pot1_to_gripper1_pos"], dtype=float)
        curr_lid = np.array(curr["lid1_to_gripper1_pos"], dtype=float)
        curr_pot = np.array(curr["pot1_to_gripper1_pos"], dtype=float)
        prev_sep = np.linalg.norm(prev_lid - prev_pot)
        curr_sep = np.linalg.norm(curr_lid - curr_pot)
        # reward if lid lifts off pot, uncovering it
        progress = (curr_sep - prev_sep) / max_disp
        return clip01(progress)

    # unrecognized effect => no shaping
    return 0.0
