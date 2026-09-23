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
    # maximum possible end-effector movement per step
    max_disp = 0.016

    def clip(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    # compute Euclidean norm
    def norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # progress = reduction in distance between gripper and drawer handle
        prev_vec = np.array(prev_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        curr_vec = np.array(current_observation_with_semantics["drawer1_handle_to_gripper1_pos"])
        prev_dist = norm(prev_vec)
        curr_dist = norm(curr_vec)
        delta_dist = prev_dist - curr_dist
        # scale by max end-effector motion, clip
        reward = delta_dist / max_disp
        return clip(reward)

    elif grounded_effect == "(open drawer1)":
        # progress = increase in extension of drawer relative to cabinet along x-axis
        prev_vec = np.array(prev_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"])
        curr_vec = np.array(current_observation_with_semantics["drawer1_cabinet_to_gripper1_pos"])
        # assume x-axis aligns with opening direction
        delta_open = (curr_vec[0] - prev_vec[0])
        # scale by max end-effector motion, clip
        reward = delta_open / max_disp
        return clip(reward)

    else:
        # no shaped reward for other effects
        return 0.0
