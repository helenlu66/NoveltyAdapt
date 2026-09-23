# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      current_observation_with_semantics:Dict[str, Union[bool, float, np.array]],
                      grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics
                                        and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have
                                           semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    # maximum end-effector displacement norm per action
    max_disp = 0.016

    # helper to clip reward
    def clip(r):
        return float(np.clip(r, -1.0, 1.0))

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # effect achieved when coffee-pod1 is gripped -> minimize distance gripper <-> coffee-pod1
        prev_d = np.linalg.norm(prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
        curr_d = np.linalg.norm(current_observation_with_semantics['coffee-pod1_to_gripper1_pos'])
        # positive reward if distance decreases
        reward = (prev_d - curr_d) / max_disp
        return clip(reward)

    elif grounded_effect == "(not (in coffee-pod1 drawer1))":
        # effect achieved when coffee-pod1 moves out of drawer -> increase distance between pod and drawer handle
        prev_vec = (prev_observation_with_semantics['coffee-pod1_to_gripper1_pos'] -
                    prev_observation_with_semantics['drawer1_handle_to_gripper1_pos'])
        curr_vec = (current_observation_with_semantics['coffee-pod1_to_gripper1_pos'] -
                    current_observation_with_semantics['drawer1_handle_to_gripper1_pos'])
        prev_d = np.linalg.norm(prev_vec)
        curr_d = np.linalg.norm(curr_vec)
        # positive reward if pod moves away from drawer handle
        reward = (curr_d - prev_d) / max_disp
        return clip(reward)

    else:
        # no shaping for other effects
        return 0.0
