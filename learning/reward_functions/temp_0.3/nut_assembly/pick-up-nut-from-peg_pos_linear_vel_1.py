# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_STEP = 0.016            # maximum possible change of the end-effector in one low-level action
EPS          = 1e-6            # small number to avoid divide-by-zero


def _l2(vec: np.ndarray) -> float:
    """Return the Euclidean length of a vector."""
    return float(np.linalg.norm(vec))


def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: semantic observation before executing the action
        current_observation_with_semantics: semantic observation after executing the action
        grounded_effect: the grounded effect whose progress should be rewarded
    Returns:
        float: dense reward in the range [-1, 1]
    '''

    # --------------- helper measurements that are reused -----------------
    # distance from gripper to the square nut handle (proxy for "nut in gripper")
    prev_nut_handle_dist = _l2(
        prev_observation_with_semantics['square-nut1_handle_to_gripper1_pos']
    )
    curr_nut_handle_dist = _l2(
        current_observation_with_semantics['square-nut1_handle_to_gripper1_pos']
    )

    # distance between the square nut and the round peg (proxy for "nut off peg")
    prev_nut_peg_vec = (
        prev_observation_with_semantics['square-nut1_to_gripper1_pos']
        - prev_observation_with_semantics['round-peg1_to_gripper1_pos']
    )
    curr_nut_peg_vec = (
        current_observation_with_semantics['square-nut1_to_gripper1_pos']
        - current_observation_with_semantics['round-peg1_to_gripper1_pos']
    )
    prev_nut_peg_dist = _l2(prev_nut_peg_vec)
    curr_nut_peg_dist = _l2(curr_nut_peg_vec)

    # ----------------------------------------------------------------------

    # Scale a signed distance change to the range [-1, 1] using the
    # maximum end-effector step as the denomi­nator.
    def _scaled_delta(delta: float) -> float:
        return float(np.clip(delta / (MAX_EE_STEP + EPS), -1.0, 1.0))

    # ----------------------------------------------------------------------
    # Effect-specific dense rewards
    # ----------------------------------------------------------------------
    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # Getting closer to the square nut’s handle is positive progress,
        # moving away is negative progress.
        distance_change = prev_nut_handle_dist - curr_nut_handle_dist
        return _scaled_delta(distance_change)

    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":
        # Increasing the nut–peg distance is progress for *removing* the nut.
        distance_change = curr_nut_peg_dist - prev_nut_peg_dist
        return _scaled_delta(distance_change)

    # ----------------------------------------------------------------------
    # Fallback: if we are asked about an unknown effect, give zero reward.
    # ----------------------------------------------------------------------
    return 0.0
