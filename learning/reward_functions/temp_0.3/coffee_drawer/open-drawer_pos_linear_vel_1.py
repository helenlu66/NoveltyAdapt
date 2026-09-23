# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_STEP = 0.016          # maximum end–effector displacement the task description guarantees
EPS            = 1e-8        # to avoid divide-by-zero


def _safe_norm(vec: np.ndarray) -> float:
    """robust euclidean length that works even when vec is None"""
    if vec is None:
        return 0.0
    return float(np.linalg.norm(vec))


def reward_shaping_fn(prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
                      current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
                      grounded_effect: str) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation *before* the action
        current_observation_with_semantics: observation *after*  the action
        grounded_effect: the specific grounded effect whose progress is being rewarded
    Returns:
        float: a dense reward in [-1, 1] measuring step-wise progress toward the effect
    '''
    # ------------------------------------------------------------------
    # utility: fetch vectors and fall back to zeros if a key is missing
    prev_handle_vec   = prev_observation_with_semantics.get(
        "drawer1_handle_to_gripper1_pos", None)
    curr_handle_vec   = current_observation_with_semantics.get(
        "drawer1_handle_to_gripper1_pos", None)

    prev_cabinet_vec  = prev_observation_with_semantics.get(
        "drawer1_cabinet_to_gripper1_pos", None)
    curr_cabinet_vec  = current_observation_with_semantics.get(
        "drawer1_cabinet_to_gripper1_pos", None)

    # convert to distances
    prev_handle_dist  = _safe_norm(prev_handle_vec)
    curr_handle_dist  = _safe_norm(curr_handle_vec)

    prev_cabinet_dist = _safe_norm(prev_cabinet_vec)
    curr_cabinet_dist = _safe_norm(curr_cabinet_vec)
    # ------------------------------------------------------------------

    # initialise reward
    shaped_reward = 0.0

    # ------------------------------------------------------------------
    # Effect 1 : the gripper ends up *exclusively occupying* the drawer handle
    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # Progress = how much closer we got to the handle in this step.
        progress = (prev_handle_dist - curr_handle_dist) / (MAX_EE_STEP + EPS)
        shaped_reward = float(np.clip(progress, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Effect 2 : the drawer becomes *open*
    elif grounded_effect == "(open drawer1)":
        # The drawer opens when the *drawer front* (along with its handle)
        # moves away from the cabinet.  We approximate this by measuring
        # how much farther the gripper is from the cabinet **while the
        # gripper is near the handle** (so that we reward pulling rather
        # than merely flying away).
        near_handle = (curr_handle_dist < 0.05)   # 5 cm proximity gate

        if near_handle:
            progress = (curr_cabinet_dist - prev_cabinet_dist) / (MAX_EE_STEP + EPS)
            shaped_reward = float(np.clip(progress, -1.0, 1.0))
        else:
            shaped_reward = 0.0   # no reward until the agent is holding the handle

    # ------------------------------------------------------------------
    return shaped_reward
