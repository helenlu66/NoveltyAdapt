# llm generated reward shaping function
from typing import *
import numpy as np

# maximum gripper movement that can happen in one low-level step (given by the
# problem statement).  We will use it to scale the dense reward into [-1, 1]
# every step.
_MAX_EE_STEP = 0.016


def _clip(x: float) -> float:
    """helper that clips any value into the interval [-1, 1]."""
    return float(np.clip(x, -1.0, 1.0))


def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str,
) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation *before* the action
        current_observation_with_semantics: observation *after* the action
        grounded_effect (str): the grounded STRIPS effect whose progress we are
                               trying to evaluate
    Returns:
        float: dense reward in [-1, 1] that reflects step-wise progress towards
               the grounded effect
    '''

    # For readability
    prev = prev_observation_with_semantics
    cur = current_observation_with_semantics

    # --------------------- 1. Coffee-pod in gripper --------------------------
    # progress criterion: the Euclidean distance between the gripper and the
    # coffee-pod should *decrease* every step until contact is made.
    if grounded_effect.strip() == "(exclusively-occupying-gripper coffee-pod1 gripper1)":
        prev_dist = np.linalg.norm(prev["coffee-pod1_to_gripper1_pos"])
        cur_dist  = np.linalg.norm(cur["coffee-pod1_to_gripper1_pos"])

        # If the pod is already very close (< 2 cm) we treat the effect as
        # achieved and hand out the maximal reward.
        if cur_dist < 0.02:
            return 1.0

        # Positive reward if distance shrinks, negative if it grows.
        # Scale by the physical upper bound of how much the distance can change
        # in one step (≤ end-effector displacement).
        improvement = prev_dist - cur_dist          # >0 means progress
        return _clip(improvement / _MAX_EE_STEP)

    # --------------- 2. Coffee-pod no longer inside the drawer ---------------
    # We cannot directly sense the logical predicate “in”, but we can infer
    # progress by how far the pod moves away from the drawer-cabinet centre.
    #
    # Both `coffee-pod1_to_gripper1_pos`  and `drawer1_cabinet_to_gripper1_pos`
    # are 3-vectors that point from the respective object to the gripper.
    # Their difference therefore equals (drawer_cabinet_pos - pod_pos), whose
    # norm is the drawer-to-pod distance.
    if grounded_effect.strip() == "(not (in coffee-pod1 drawer1))":
        prev_vec_pod     = prev["coffee-pod1_to_gripper1_pos"]
        prev_vec_drawer  = prev["drawer1_cabinet_to_gripper1_pos"]
        cur_vec_pod      = cur["coffee-pod1_to_gripper1_pos"]
        cur_vec_drawer   = cur["drawer1_cabinet_to_gripper1_pos"]

        prev_dist = np.linalg.norm(prev_vec_pod - prev_vec_drawer)
        cur_dist  = np.linalg.norm(cur_vec_pod  - cur_vec_drawer)

        # If the pod is more than ~8 cm away from the cabinet we consider it
        # definitely “out of the drawer”.
        if cur_dist > 0.08:
            return 1.0

        # Reward increases when the pod moves away from the drawer.
        improvement = cur_dist - prev_dist          # >0 means progress
        return _clip(improvement / _MAX_EE_STEP)

    # -------------------- 3. Unknown grounded effect -------------------------
    # If the caller supplies an effect we do not know how to shape for, we
    # return a neutral reward.
    return 0.0
