# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_STEP = 0.016      # stated in the prompt
EPS          = 1e-6      # numerical safety


def _clamp(v : float, low : float = -1.0, high : float = 1.0) -> float:
    return float(max(low, min(high, v)))


def reward_shaping_fn(
        prev_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
        current_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
        grounded_effect : str
) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation before the action
        current_observation_with_semantics: observation after the action
        grounded_effect (str): the grounded effect whose progress we are rewarding
    Returns:
        float between -1 and 1
    '''

    #  ------------------------------------------------------------------
    #  Utility lambdas
    #  ------------------------------------------------------------------
    def dist_handle_to_gripper(obs : Dict[str, Any]) -> float:
        """      Euclidean distance   between drawer1-handle and gripper1."""
        return float(np.linalg.norm(obs['drawer1_handle_to_gripper1_pos']))

    def dist_cabinet_to_handle(obs : Dict[str, Any]) -> float:
        """      Distance between cabinet frame and handle, obtained only with
           quantities that are present in the observation dictionaries:

                 p_cabinet→handle = p_cabinet→gripper  –  p_handle→gripper
        """
        cabinet_to_gripper = obs['drawer1_cabinet_to_gripper1_pos']
        handle_to_gripper  = obs['drawer1_handle_to_gripper1_pos']
        cabinet_to_handle  = handle_to_gripper - cabinet_to_gripper
        return float(np.linalg.norm(cabinet_to_handle))

    # --------------------------------------------------------------------
    # Grounded effect specific progress calculation
    # --------------------------------------------------------------------
    if grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # Getting the gripper closer to the handle (until it grasps it)
        prev_d = dist_handle_to_gripper(prev_observation_with_semantics)
        curr_d = dist_handle_to_gripper(current_observation_with_semantics)

        # Positive reward when distance gets smaller, negative otherwise
        progress = (prev_d - curr_d) / (MAX_EE_STEP + EPS)      # normalise
        return _clamp(progress)

    elif grounded_effect == "(open drawer1)":
        # The drawer is open when its handle is further away from the cabinet.
        prev_d = dist_cabinet_to_handle(prev_observation_with_semantics)
        curr_d = dist_cabinet_to_handle(current_observation_with_semantics)

        # Positive reward when the handle moves away from the cabinet
        progress = (curr_d - prev_d) / (MAX_EE_STEP + EPS)
        return _clamp(progress)

    # --------------------------------------------------------------------
    # Unknown effect –> no shaping
    # --------------------------------------------------------------------
    return 0.0