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
    # maximum possible end-effector travel distance in one step
    max_disp = 0.016

    # parse out the relative positions
    prev_box_to_grip   = np.array(prev_observation_with_semantics["box1_to_gripper1_pos"])
    curr_box_to_grip   = np.array(current_observation_with_semantics["box1_to_gripper1_pos"])
    prev_pod_to_grip   = np.array(prev_observation_with_semantics["coffee-pod1_to_gripper1_pos"])
    curr_pod_to_grip   = np.array(current_observation_with_semantics["coffee-pod1_to_gripper1_pos"])

    # distances gripper<->coffee-pod
    prev_dist_pod_grip = np.linalg.norm(prev_pod_to_grip)
    curr_dist_pod_grip = np.linalg.norm(curr_pod_to_grip)

    # distances box<->coffee-pod estimated via their relative vectors to the gripper
    prev_dist_box_pod = np.linalg.norm(prev_box_to_grip - prev_pod_to_grip)
    curr_dist_box_pod = np.linalg.norm(curr_box_to_grip - curr_pod_to_grip)

    if grounded_effect == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        # we want the coffee-pod to get closer to the gripper
        # positive reward if distance decreases
        reward = (prev_dist_pod_grip - curr_dist_pod_grip) / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    elif grounded_effect == "(not (in coffee-pod1 box1))":
        # we want the coffee-pod to move out of the box
        # positive reward if box-pod separation increases
        reward = (curr_dist_box_pod - prev_dist_box_pod) / max_disp
        return float(np.clip(reward, -1.0, 1.0))

    else:
        # no shaping for other effects
        return 0.0
