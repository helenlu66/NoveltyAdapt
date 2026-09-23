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
        prev_observation_with_semantics: the previous observation before taking the action,
            keys have semantics and values are numeric or boolean.
        current_observation_with_semantics: the current observation after taking the action,
            keys have semantics and values are numeric or boolean.
        grounded_effect (str): the grounded effect of the operator whose progress we measure.
    Returns:
        float: a dense reward in [-1, 1] estimating progress toward the grounded effect.
    '''
    # Maximum possible change in end-effector position per action
    max_disp = 0.016
    
    prev_obs = prev_observation_with_semantics
    curr_obs = current_observation_with_semantics

    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # we expect the nut's handle to get closer to the gripper
        prev_dist = np.linalg.norm(prev_obs['square-nut1_handle_to_gripper1_pos'])
        curr_dist = np.linalg.norm(curr_obs['square-nut1_handle_to_gripper1_pos'])
        # positive reward when distance decreases
        delta = prev_dist - curr_dist
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # we expect the nut to move away from the peg
        prev_vec = (prev_obs['square-nut1_to_gripper1_pos'] -
                    prev_obs['round-peg1_to_gripper1_pos'])
        curr_vec = (curr_obs['square-nut1_to_gripper1_pos'] -
                    curr_obs['round-peg1_to_gripper1_pos'])
        prev_sep = np.linalg.norm(prev_vec)
        curr_sep = np.linalg.norm(curr_vec)
        # positive reward when separation increases
        delta = curr_sep - prev_sep
        reward = delta / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    # if we have no matching grounded effect, no shaping reward
    return 0.0
