# llm generated reward shaping function
from typing import *
import numpy as np

MAX_EE_DISPLACEMENT_PER_STEP = 0.016  # given in the prompt


def _clip_progress(raw_delta: float) -> float:
    """
    Maps a change in a distance measure to the range [-1, 1] by dividing
    by the maximum possible change of the end-effector in one low-level step
    and clipping.
    """
    return float(np.clip(raw_delta / MAX_EE_DISPLACEMENT_PER_STEP, -1.0, 1.0))


def reward_shaping_fn(
    prev_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    current_observation_with_semantics: Dict[str, Union[bool, float, np.ndarray]],
    grounded_effect: str,
) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation before the action
        current_observation_with_semantics: observation after the action
        grounded_effect: one of the grounded effects whose progress we score
    Returns:
        float in [-1, 1]: positive means progress towards satisfying the effect,
                          negative means regress.
    '''
    # Convenience handles – will raise a clear error if a key is missing
    prev = prev_observation_with_semantics
    cur  = current_observation_with_semantics

    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        # Measure distance between lid and gripper
        prev_dist = np.linalg.norm(prev["lid1_to_gripper1_pos"])
        cur_dist  = np.linalg.norm(cur["lid1_to_gripper1_pos"])

        # A decrease in the distance means the lid is moving into the gripper
        progress = _clip_progress(prev_dist - cur_dist)
        return progress

    elif grounded_effect == "(not (ontop lid1 pot1))":
        # Distance between lid and pot
        prev_vec = prev["lid1_to_gripper1_pos"] - prev["pot1_to_gripper1_pos"]
        cur_vec  = cur ["lid1_to_gripper1_pos"] - cur ["pot1_to_gripper1_pos"]

        prev_dist = np.linalg.norm(prev_vec)
        cur_dist  = np.linalg.norm(cur_vec)

        # An increase in this distance breaks the “ontop” relation
        progress = _clip_progress(cur_dist - prev_dist)
        return progress

    elif grounded_effect == "(not (covered pot1))":
        # Being “covered” by the lid is also broken when the lid moves away.
        # We reuse the same distance measure as above.
        prev_vec = prev["lid1_to_gripper1_pos"] - prev["pot1_to_gripper1_pos"]
        cur_vec  = cur ["lid1_to_gripper1_pos"] - cur ["pot1_to_gripper1_pos"]

        prev_dist = np.linalg.norm(prev_vec)
        cur_dist  = np.linalg.norm(cur_vec)

        progress = _clip_progress(cur_dist - prev_dist)
        return progress

    else:
        # Unknown effect – give no shaping
        return 0.0
