# llm generated reward shaping function
from typing import *
import numpy as np


_MAX_EE_STEP = 0.016      # maximum end-effector displacement per low-level action
_EPS            = 1e-8    # for numerical stability


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str,
) -> float:
    '''
    Args
    ----
        prev_observation_with_semantics : observation *before* taking the action
        current_observation_with_semantics : observation *after* taking the action
        grounded_effect : the grounded effect whose progress we want to reward

    Returns
    -------
        float : dense reward in the range [-1, 1]
    '''

    # ---------- effect 1: (exclusively-occupying-gripper square-nut1 gripper1) ----------
    # Heuristic: the nut should be as close as possible to the gripper’s fingers/handle.
    if grounded_effect.strip() == "(exclusively-occupying-gripper square-nut1 gripper1)":
        # Distance from the nut handle to the gripper
        prev_d = np.linalg.norm(
            prev_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        )
        cur_d = np.linalg.norm(
            current_observation_with_semantics["square-nut1_handle_to_gripper1_pos"]
        )

        # Positive progress = the nut became closer to the gripper
        progress = (prev_d - cur_d) / (_MAX_EE_STEP + _EPS)
        return _clamp(progress)

    # ---------- effect 2: (not (on-peg square-nut1 round-peg1)) ----------
    # Heuristic: the nut should get farther away from the peg centre.
    elif grounded_effect.strip() == "(not (on-peg square-nut1 round-peg1))":
        # Vector from peg to nut (nut_rel_peg = nut_rel_gripper - peg_rel_gripper)
        prev_vec = (
            prev_observation_with_semantics["square-nut1_to_gripper1_pos"]
            - prev_observation_with_semantics["round-peg1_to_gripper1_pos"]
        )
        cur_vec = (
            current_observation_with_semantics["square-nut1_to_gripper1_pos"]
            - current_observation_with_semantics["round-peg1_to_gripper1_pos"]
        )

        prev_d = np.linalg.norm(prev_vec)
        cur_d = np.linalg.norm(cur_vec)

        # Positive progress = the nut moved farther from the peg
        progress = (cur_d - prev_d) / (_MAX_EE_STEP + _EPS)
        return _clamp(progress)

    # ---------- unknown grounded effect ----------
    else:
        # No shaping for effects we do not recognise
        return 0.0
