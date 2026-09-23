linear_vel_reward_shaping_prompt = """Generate a dense reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary for the reset environment is as follows. In the reset environment observation, the gripper is open, above the center of the table, and far away from any object. No effect has been achieved in the reset environment:
```
{observation_with_semantics}
```
You should first check what semantic information is in the observations. You should then write code to calculate the progress made towards the grounded effect at any step based on how the current observation changed from the previous observation. Assume that the maximum displacement of the end effector is 0.016 during each action. DO NOT MAKE UP NUMBERS. Do not use keys that are not shown in the observation dictionaries. You can calculate the distance between two objects by calculating the difference between their respective relative distances to the gripper. Wrap your reward function in a code block (triple backticks) in the following format
The template:
```
# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],current_observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between -1 and 1 based on the grounded effect
    '''
    ...
    if grounded_effect == "(effect1)":
        # calculate pogress between -1 and 1
        ...
    elif grounded_effect == "(effect2)":
        # calculate progress between -1 and 1
        ...
    ...
```
"""
different_candidate_prompt = """Existing reward shaping function candidates for the grounded operator:
{existing_candidates_str}
Try to generate a reward shaping function candidate that is different from the existing candidates for the same grounded operator. The different candidate should have semantically different progress calculation from the existing candidates for at least one of the grounded effects/subgoals."""

linear_dist_reward_shaping_prompt = """Generate a dense reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary for the reset environment is as follows. In the reset environment observation, the gripper is open, above the center of the table, and far away from any object. No effect has been achieved in the reset environment:
```
{observation_with_semantics}
```
You should first check what semantic information is in the observations. You should write code to calculate the linear progress towards the grounded effect based on the values of the current observation. Do not make up numbers. Do not use keys that are not shown in the observation dictionaries. You can calculate the distance between two objects by calculating the difference between their respective relative distances to the gripper. Wrap your reward function in a code block (triple backticks) in the following format.
The template:
```
# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the linear reward between 0 and 1 based on the grounded effect
    '''
    ...
    if grounded_effect == "(effect1)":
        # calculate percent pogress between 0 and 1
        ...
    elif grounded_effect == "(effect2)":
        # calculate percent progress between 0 and 1
        ...
    ...
```
"""

 
exponential_vel_reward_shaping_prompt = """Generate a dense exponential reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary for the reset environment is as follows. In the reset environment observation, the gripper is open, above the center of the table, and far away from any object. No effect has been achieved in the reset environment:
```
{observation_with_semantics}
```
You should first check what semantic information is in the observations. You should then write code to calculate progress towards the grounded effect based on the information in the initial, the previous and the current observations. Assume that the maximum possible displacement of the end effector is 0.016 during each action. Do not make up numbers. Do not use keys that are not shown in the observation dictionaries. You can calculate the distance between two objects by calculating the difference between their respective relative distances to the gripper. Wrap your reward function in a code block (triple backticks) in the following format
The template:
```
# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(prev_observation_with_semantics:Dict[str, Union[bool, float, np.array]],current_observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        prev_observation_with_semantics: the previous observation before taking the action in which the keys have semantics and the values are arrays of numeric values before taking the action
        current_observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the eponential reward between 0 and 1 based on the grounded effect
    '''
    ...
    if grounded_effect == "(effect1)":
        # calculate percent pogress
        ...
        return (np.exp(progress) - 1) / (np.exp(1) - 1)
    elif grounded_effect == "(effect2)":
        # calculate percent progress
        ...
        return (np.exp(progress) - 1) / (np.exp(1) - 1)
    ...
```
"""

# placeholder prompt for the LLM to generate a reward function. 
exponential_dist_reward_shaping_prompt = """Generate a dense exponential reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary for the reset environment is as follows. In the reset environment observation, the gripper is open, above the center of the table, and far away from any object. No effect has been achieved in the reset environment:
```
{observation_with_semantics}
```
You should first check what semantic information is in the observations. You should write code to calculate the linear progress towards the grounded effect based on the values of the current observation. Do not make up numbers. Do not use keys that are not shown in the observation dictionaries. You can calculate the distance between two objects by calculating the difference between their respective relative distances to the gripper. Wrap your reward function in a code block (triple backticks) in the following format.
The template:
```
# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics: the current observation after taking the action in which the keys have semantics and the values are arrays of numeric values after taking the action
        grounded_effect (str): the grounded effect of the operator whose progress we are trying to measure
    Returns:
        float: the eponential reward between 0 and 1 based on the grounded effect
    '''
    ...
    if grounded_effect == "(effect1)":
        # calculate percent pogress between 0 and 1
        ...
        return (np.exp(progress) - 1) / (np.exp(1) - 1)
    elif grounded_effect == "(effect2)":
        # calculate percent progress between 0 and 1
        ...
        return (np.exp(progress) - 1) / (np.exp(1) - 1)
    ...
```
"""