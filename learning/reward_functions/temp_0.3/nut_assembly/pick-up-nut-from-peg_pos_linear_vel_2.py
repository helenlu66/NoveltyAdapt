# llm generated reward shaping function
from typing import Dict, Union
import numpy as np

# upper-bound on how much any object can move in one low-level action
_MAX_STEP = 0.016          # supplied by the task description
_EPS        = 1e-8          # small value to avoid divide-by-zero


def _safe_norm(vec: np.ndarray) -> float:
    """ Robust L2-norm that works on python lists and avoids division by zero."""
    return float(np.linalg.norm(np.asarray(vec)))


def reward_shaping_fn(
    prev_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect : str
) -> float:
    '''
    Dense shaping signal for the two effects of the grounded operator

        pick-up-nut-from-peg(gripper1, round-peg1, square-nut1)

    Keys that are available inside an observation:
        • 'square-nut1_handle_to_gripper1_pos' : 3-D vector from the square nut’s handle to the gripper
        • 'round-peg1_handle_to_gripper1_pos'  : 3-D vector from the round peg’s handle to the gripper
        • (plus some others that are not required here)

    Shape of the reward:
        – it is the signed progress that happened between prev-obs and curr-obs
        – it is clipped to the interval [-1 , 1]
    '''

    # ------------------------------------------------------------------
    # effect 1 : (exclusively-occupying-gripper square-nut1 gripper1)
    # ------------------------------------------------------------------
    # Intuition : the gripper must close around the nut handle ⇒ the distance
    #             between the nut and the gripper should decrease.
    if grounded_effect == "(exclusively-occupying-gripper square-nut1 gripper1)":

        prev_dist  = _safe_norm(prev_observation_with_semantics[
                                'square-nut1_handle_to_gripper1_pos'])
        curr_dist  = _safe_norm(current_observation_with_semantics[
                                'square-nut1_handle_to_gripper1_pos'])

        # Positive reward if the distance became smaller, negative otherwise
        delta      = prev_dist - curr_dist          # improvement (>0) or regression (<0)
        reward     = np.clip(delta / _MAX_STEP, -1.0, 1.0)
        return float(reward)

    # ------------------------------------------------------------------
    # effect 2 : (not (on-peg square-nut1 round-peg1))
    # ------------------------------------------------------------------
    # Intuition : the nut must separate from the peg ⇒ the distance
    #             between nut and peg should increase.
    elif grounded_effect == "(not (on-peg square-nut1 round-peg1))":

        prev_nut_to_gripper  = np.asarray(prev_observation_with_semantics[
                                         'square-nut1_to_gripper1_pos'])
        curr_nut_to_gripper  = np.asarray(current_observation_with_semantics[
                                         'square-nut1_to_gripper1_pos'])

        prev_peg_to_gripper  = np.asarray(prev_observation_with_semantics[
                                         'round-peg1_to_gripper1_pos'])
        curr_peg_to_gripper  = np.asarray(current_observation_with_semantics[
                                         'round-peg1_to_gripper1_pos'])

        # distance nut ↔ peg  (nut - peg)
        prev_dist = _safe_norm(prev_nut_to_gripper - prev_peg_to_gripper)
        curr_dist = _safe_norm(curr_nut_to_gripper - curr_peg_to_gripper)

        # Positive reward if the distance became larger, negative otherwise
        delta     = curr_dist - prev_dist           # improvement (>0) or regression (<0)
        reward    = np.clip(delta / _MAX_STEP, -1.0, 1.0)
        return float(reward)

    # ------------------------------------------------------------------
    # Unsupported effect ‑> no shaping signal
    # ------------------------------------------------------------------
    else:
        return 0.0
