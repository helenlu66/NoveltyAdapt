# llm generated reward shaping function
from typing import *
import numpy as np

"""
The only numeric information we receive from the environment is the relative
position vectors contained in the observation dictionaries.  We therefore have
to measure “progress” towards each literal that appears in the grounded
operator by reasoning exclusively over those vectors.

Keys that are available
───────────────────────
gripper1_pos                  : absolute EE position in world coordinates
pot1_to_gripper1_pos          : pot   position expressed in the gripper frame
lid1_to_gripper1_pos          : lid   position expressed in the gripper frame
lid1_handle_to_gripper1_pos   : lid–handle position expressed in the gripper frame

How the three grounded effects can be detected
──────────────────────────────────────────────
1.  (exclusively-occupying-gripper lid1 gripper1)
    – The gripper must be holding the lid’s handle, which can only happen
      when the handle is within a very small radius of the gripper TCP.
      We measure the Euclidean norm of  lid1_handle_to_gripper1_pos .

2.  (not (ontop lid1 pot1))
    – The lid must no longer be centred on the pot.  
      We compute lid-to-pot displacement in the horizontal (x-y) plane
      by subtracting the two “*_to_gripper1_pos” vectors and then taking
      the 2-norm of the first two components.

3.  (not (covered pot1))
    – Using the same difference vector as above we look at the absolute z–component.

Dense reward design
───────────────────
For every effect we award
    progress = (prev_distance – current_distance) / 0.016
where 0.016 m is the maximum end-effector displacement per step that the
user specified.  This normalises the *best possible* per-step improvement
to +1 and the *worst possible* degradation to -1. 

All rewards are finally clipped to the interval [-1, 1].
"""

def reward_shaping_fn(
        prev_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
        current_observation_with_semantics : Dict[str, Union[bool, float, np.ndarray]],
        grounded_effect : str) -> float:
    '''
    Args:
        prev_observation_with_semantics: observation before the action
        current_observation_with_semantics: observation after the action
        grounded_effect (str): the grounded literal whose progress we score
    Returns:
        float: a reward in the range [-1, 1]
    '''

    # helper to stay inside [-1,1]
    def clip(r):                    # type: (float) -> float
        return float(np.clip(r, -1., 1.))

    # distance gripper → lid-handle
    if grounded_effect == "(exclusively-occupying-gripper lid1 gripper1)":
        prev_d = np.linalg.norm(prev_observation_with_semantics["lid1_handle_to_gripper1_pos"])
        curr_d = np.linalg.norm(current_observation_with_semantics["lid1_handle_to_gripper1_pos"])

        progress = (prev_d - curr_d) / 0.016
        return clip(progress)

    # horizontal displacement between lid and pot
    elif grounded_effect == "(not (ontop lid1 pot1))":
        prev_vec = prev_observation_with_semantics["lid1_to_gripper1_pos"]   \
                 - prev_observation_with_semantics["pot1_to_gripper1_pos"]
        curr_vec = current_observation_with_semantics["lid1_to_gripper1_pos"] \
                 - current_observation_with_semantics["pot1_to_gripper1_pos"]

        prev_h = np.linalg.norm(prev_vec[:2])   # xy-plane separation
        curr_h = np.linalg.norm(curr_vec[:2])

        progress = (curr_h - prev_h) / 0.016    # larger is good
        return clip(progress)

    # vertical separation lid↑ – pot
    elif grounded_effect == "(not (covered pot1))":
        prev_vec = prev_observation_with_semantics["lid1_to_gripper1_pos"]   \
                 - prev_observation_with_semantics["pot1_to_gripper1_pos"]
        curr_vec = current_observation_with_semantics["lid1_to_gripper1_pos"] \
                 - current_observation_with_semantics["pot1_to_gripper1_pos"]

        prev_z = abs(prev_vec[2])
        curr_z = abs(curr_vec[2])

        progress = (curr_z - prev_z) / 0.016    # positive progress if lid moves upward
        return clip(progress)

    else:
        # Unknown literal – give dense 0 so as not to distort learning
        return 0.0
