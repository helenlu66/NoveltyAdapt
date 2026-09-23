# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_DISPLACEMENT_PER_STEP = 0.016  # given in the task statement


def _euclidean(a: np.ndarray) -> float:
    """Length of a vector."""
    return float(np.linalg.norm(a))


def _clip_to_unit(x: float) -> float:
    """Keep value in [-1, 1]."""
    return float(np.clip(x, -1.0, 1.0))


def reward_shaping_fn(
        prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
        current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
        grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation *before* the agent’s action
        current_observation_with_semantics: observation *after* the agent’s action
        grounded_effect (str): the grounded effect whose progress we want to
                               translate into a dense reward
    Returns:
        float: a reward in [-1, 1] that is proportional to the progress towards
               satisfying `grounded_effect`
    '''
    # convenience handles -------------------------------------------------------
    prev = prev_observation_with_semantics
    cur  = current_observation_with_semantics

    ###########################################################################
    # 1) (exclusively-occupying-gripper drawer1 gripper1)
    #
    #    A reasonable proxy for “gripper is exclusively occupying drawer1” is
    #    the distance between the gripper and the drawer handle.  As this
    #    distance decreases the agent is getting closer to grasping / occupying
    #    the drawer handle.  Because the drawer handle is fixed to the drawer,
    #    we can compute that distance directly from the provided
    #    `*_handle_to_gripper1_pos` vector.
    #
    #    Reward:  (prev_dist – cur_dist) / MAX_EE_DISPLACEMENT_PER_STEP
    #
    #    •  Positive reward when the distance shrinks
    #    •  Negative reward when the distance grows
    #    •  The quotient is clipped to [-1, 1]
    ###########################################################################
    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        prev_dist = _euclidean(prev["drawer1_handle_to_gripper1_pos"])
        cur_dist  = _euclidean(cur["drawer1_handle_to_gripper1_pos"])

        # progress is the amount we reduced the distance, normalised
        progress  = (prev_dist - cur_dist) / MAX_EE_DISPLACEMENT_PER_STEP
        return _clip_to_unit(progress)

    ###########################################################################
    # 2) (open drawer1)
    #
    #    Opening a drawer means pulling the drawer’s body away from its cabinet.
    #    While we do not observe the drawer’s internal state directly, we *do*
    #    observe the relative gripper-to-cabinet vector:
    #
    #        drawer1_cabinet_to_gripper1_pos
    #
    #    When the drawer is successfully opened *and the agent is pulling it*,
    #    the cabinet‐to‐gripper distance along the drawer’s sliding axis
    #    (usually the Y-axis in these tasks) *decreases in magnitude*: the
    #    gripper moves closer to the cabinet’s front face as the drawer
    #    translates outwards.
    #
    #    Therefore, we treat a reduction in the absolute Y component of that
    #    vector as progress towards the drawer being open.
    #
    #    Reward:  (|prev_y| – |cur_y|) / MAX_EE_DISPLACEMENT_PER_STEP
    #             clipped to [-1, 1]
    ###########################################################################
    elif grounded_effect == "(open drawer1)":
        prev_y = float(abs(prev["drawer1_cabinet_to_gripper1_pos"][1]))
        cur_y  = float(abs(cur ["drawer1_cabinet_to_gripper1_pos"][1]))

        progress = (prev_y - cur_y) / MAX_EE_DISPLACEMENT_PER_STEP
        return _clip_to_unit(progress)

    # -------------------------------------------------------------------------
    # Unknown grounded effect
    # -------------------------------------------------------------------------
    else:
        # If we do not have a specific shaping rule for the requested effect,
        # return neutral reward.
        return 0.0
