# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_STEP = 0.016          # stated maximum end–effector displacement / env-step
                             # we use it to normalise every shaped reward to [-1 , 1]

def _clip(x: float) -> float:
    """Clip the shaped reward to [-1 , 1]."""
    return float(np.clip(x, -1.0, 1.0))


def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation *before* an action
        current_observation_with_semantics: observation *after* the action
        grounded_effect:   one of the grounded effects whose progress is to be
                           estimated:
                               "(exclusively-occupying-gripper lid1 gripper1)"
                               "(not (ontop lid1 pot1))"
                               "(not (covered pot1))"
    Returns:
        float: dense reward in [-1 , 1] representing step-wise progress
    '''
    # Helper shortcuts (they always exist in the provided observations)
    p_lid_vec_prev   = prev_observation_with_semantics['lid1_to_gripper1_pos']
    p_lid_vec_curr   = current_observation_with_semantics['lid1_to_gripper1_pos']
    p_lid_handle_prev = prev_observation_with_semantics['lid1_handle_to_gripper1_pos']
    p_lid_handle_curr = current_observation_with_semantics['lid1_handle_to_gripper1_pos']
    p_pot_vec_prev   = prev_observation_with_semantics['pot1_to_gripper1_pos']
    p_pot_vec_curr   = current_observation_with_semantics['pot1_to_gripper1_pos']

    # ------------------------------------------------------------------
    # (1) lid exclusively held by the gripper  →  minimise distance
    # ------------------------------------------------------------------
    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # we measure how much closer the gripper came to the lid’s handle
        dist_prev  = np.linalg.norm(p_lid_handle_prev)
        dist_curr  = np.linalg.norm(p_lid_handle_curr)

        # positive reward if the distance decreased, negative if it increased
        progress   = (dist_prev - dist_curr) / MAX_EE_STEP
        return _clip(progress)

    # ------------------------------------------------------------------
    # (2) lid no longer on top of the pot  →  maximise lid–pot separation
    # ------------------------------------------------------------------
    elif grounded_effect == "(not (ontop lid1 pot1))":
        # lid–pot relative vectors (through the common gripper frame)
        #   vector from pot to lid  =  (lid→gripper) - (pot→gripper)
        pot_to_lid_prev = p_lid_vec_prev - p_pot_vec_prev
        pot_to_lid_curr = p_lid_vec_curr - p_pot_vec_curr

        dist_prev = np.linalg.norm(pot_to_lid_prev)
        dist_curr = np.linalg.norm(pot_to_lid_curr)

        # positive reward when the separation grows
        progress  = (dist_curr - dist_prev) / MAX_EE_STEP
        return _clip(progress)

    # ------------------------------------------------------------------
    # (3) pot is not covered  →  same geometric signal as (2)
    # ------------------------------------------------------------------
    elif grounded_effect == "(not (covered pot1))":
        # “pot not covered” is achieved through the same physical change:
        # increasing lid–pot distance, hence we reuse the same metric.
        pot_to_lid_prev = p_lid_vec_prev - p_pot_vec_prev
        pot_to_lid_curr = p_lid_vec_curr - p_pot_vec_curr

        dist_prev = np.linalg.norm(pot_to_lid_prev)
        dist_curr = np.linalg.norm(pot_to_lid_curr)

        progress  = (dist_curr - dist_prev) / MAX_EE_STEP
        return _clip(progress)

    # ------------------------------------------------------------------
    # default – no shaped signal for unknown effects
    # ------------------------------------------------------------------
    return 0.0
