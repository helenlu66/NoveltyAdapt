# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script that offers an easy way to test random actions in a MimicGen environment.
Similar to the demo_random_action.py script from robosuite.
"""
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from detection.coffee_box_detector import CoffeeBoxDetector
from detection.coffee_drawer_detector import CoffeeDrawerDetector
from detection.cleanup_detector import CleanupDetector
from detection.nut_assembly_detector import NutAssemblyDetector
from detection.kitchen_detector import KitchenDetector
from robosuite.wrappers import GymWrapper
from utils import *
from learning.learner import *
from learning.custom_gym_wrapper import *
import numpy as np
import mimicgen

config = load_config("config.yaml")

def choose_mimicgen_environment(args):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """

    # try to import robosuite task zoo to include those envs in the robosuite registry
    try:
        import robosuite_task_zoo
    except ImportError:
        print("Failed to import robosuite_task_zoo")
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # keep only envs that correspond to the different reset distributions from the paper
    # only keep envs that end with "Novelty"
    envs = [env for env in all_envs if env.lower()[-7:] == "novelty"]

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        if args.domain.replace('_', '') in env.lower() or args.domain in env.lower():
            print("[{}] {}".format(k, env))
    print()
    try:
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    print("Chosen environment: {}\n".format(envs[k]))
    # Return the chosen environment name
    return envs[k]

def choose_op_wrap(domain, env):
    """Prints out the previously non-existing operators in the plan and returns
       the selected operator to wrap

    Args:
        domain (str): The environment domain
        env (MujocoEnv): The environment to wrap

    Returns:
        fs.Action: Chosen operator to wrap
    """
    plan = load_plan(config['planning'][domain])
    previous_executors = {}
    # Select operator to wrap
    print("Here is a list of operators in the plan:\n")
    for i, op in enumerate(plan):
        # check if the operator has a predefined executor. 
        # If it does, that means it does not need to be learned
        executor = load_executor(config, domain, op)
        if executor is None:
            print("[{}] {}".format(i, op))
            break
        else:
            previous_executors[op] = executor
            print(f"[{i}] Operator {op} has a predefined executor.")
    print()
    try:
        s = input("Choose an operator to wrap " + "(enter a number from 0 to {}): ".format(len(plan) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(plan))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(plan[k]))
    chosen_op = plan[k]
    gym_env = GymWrapper(env)
    return CollisionAblatedOperatorWrapper(
        env=gym_env,
        grounded_operator=chosen_op,
        args = args,
        config=config,
        curr_subgoal=None
    )

def choose_subgoal(op_wrap:OperatorWrapper) -> OperatorWrapper:
    """Prints out the subgoals in the grounded operator and returns the selected
       subgoal

    Args:
        op_wrap (OperatorWrapper): The OperatorWrapper whose subgoal needs to be
        set

    Returns:
        the same OperatorWrapper with the selected subgoal set
    """
    # select the subgoal for the operator
    print("Here is a list of subgoals in the grounded operator:\n")
    for i, effect in enumerate(op_wrap.grounded_operator.effects):
        print(f"[{i}] {effect.pddl_repr()}")
    try:
        s = input("Choose a subgoal " + "(enter a number from 0 to {}): ".format(len(op_wrap.grounded_operator.effects) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(op_wrap.grounded_operator.effects))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(op_wrap.grounded_operator.effects[k].pddl_repr()))
    op_wrap.curr_subgoal = op_wrap.grounded_operator.effects[k]
    return op_wrap

def choose_reward_shaping_fn(op_wrap:OperatorWrapper) -> OperatorWrapper:
    """Prints out the reward shaping functions and returns the selected reward shaping function

    Args:
        op_wrap (OperatorWrapper): The OperatorWrapper whose subgoal reward shaping functions need to be se

    Returns:
        the same OperatorWrapper with the selected reward shaping functions set
    """
    # select the reward shaping file for each effect in the grounded operator
    print("Here is a list of reward shaping functions for each effect in the grounded operator:\n")
    op_name, _ = extract_name_params_from_grounded(op_wrap.grounded_operator.ident())
    for effect in op_wrap.grounded_operator.effects:
        print(f"Reward shaping functions for {effect.pddl_repr()}:")
        for i, file_name in enumerate(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")):
            if "__pycache__" in file_name:
                continue
            if op_wrap.args.rw_type and op_wrap.args.rw_type not in file_name:
                continue
            if op_name in file_name:
                print(f"[{i}] {file_name}")
        try:
            s = input("Choose a reward shaping function for the effect " + "(enter a number from 0 to {}): ".format(len(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")) - 1))
            # parse input into a number within range
            k = min(max(int(s), 0), len(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")))
        except:
            k = 0
            print("Input is not valid. Use {} by default.\n".format(os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")[k]))
        # drop the .py extension
        reward_shaping_fn_file_name = os.listdir(f"learning/reward_functions/{op_wrap.args.domain}")[k].split(".")[0]
    
        reward_shaping_fn_module = importlib.import_module(f"learning.reward_functions.{op_wrap.args.domain}.{reward_shaping_fn_file_name}")
        reward_shaping_func = getattr(reward_shaping_fn_module, 'reward_shaping_fn')
        op_wrap.set_subgoal_reward_shaping_fn(effect, reward_shaping_func)
        print()
    return op_wrap


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a baseline agent')
    parser.add_argument('--domain', type=str, default='nut_assembly', help='the domain')
    parser.add_argument('--rl_algorithm', type=str, default='PPO', help='the RL algorithm to use')
    parser.add_argument('--rw_type', choices=['pos_linear_dist', 'pos_linear_vel', 'pos_exp_dist_test', 'pos_exp_dist', 'pos_exp_vel','neg_exp_dist', 'neg_exp_vel'], default='pos_linear_vel', help='the type of reward shaping function to use')
    parser.add_argument('--termination_type', choices=['on_first_subgoal_success', 'on_all_subgoal_success', 'no_termination', 'on_non_terminal_subgoal_success'], default='on_first_subgoal_success', help='whether to terminate the episode on success. If false, the subgoal bonus will be smaller as powers of 10.')
    parser.add_argument('--ep_len', type=int, default=100)
    args = parser.parse_args()
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment
    options["env_name"] = choose_mimicgen_environment(args)

    # Choose robot
    options["robots"] = choose_robots(exclude_bimanual=True)

    # Load the desired controller
    if 'cleanup' in options['env_name'].lower():
        options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
    else:
        options["controller_configs"] = load_controller_config(default_controller="OSC_POSITION")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
        hard_reset=False,
    )
    wrapped_env = env #VisualizationWrapper(env, indicator_configs=None)

    if options["env_name"] == "Coffee_Box_Novelty":
        args.domain = "coffee_box"
        detector = CoffeeBoxDetector(wrapped_env)
        if "Pre_Novelty" not in options["env_name"]:
            wrapped_env = choose_op_wrap("coffee_box", wrapped_env)
            wrapped_env = choose_subgoal(wrapped_env)
            wrapped_env = choose_reward_shaping_fn(wrapped_env)
    elif options["env_name"][:2] == "Co":
        args.domain = "coffee_drawer"
        detector = CoffeeDrawerDetector(wrapped_env)
        if "Pre_Novelty" not in options["env_name"]:
            wrapped_env = choose_op_wrap("coffee_drawer", wrapped_env)
            wrapped_env = choose_subgoal(wrapped_env)
            wrapped_env = choose_reward_shaping_fn(wrapped_env)
    elif options["env_name"][:2] == "Cu":
        args.domain = "cleanup"
        detector = CleanupDetector(wrapped_env)
        if "Pre_Novelty" not in options["env_name"]:
            wrapped_env = choose_op_wrap("cleanup", wrapped_env)
            wrapped_env = choose_subgoal(wrapped_env)
            wrapped_env = choose_reward_shaping_fn(wrapped_env)
    elif options["env_name"][:2] == "Nu":
        args.domain = "nut_assembly"
        detector = NutAssemblyDetector(wrapped_env)
        if "Pre_Novelty" not in options["env_name"]:
            wrapped_env = choose_op_wrap("nut_assembly", wrapped_env)
            wrapped_env = choose_subgoal(wrapped_env)
            wrapped_env = choose_reward_shaping_fn(wrapped_env)
    elif options["env_name"][:2] == "Ki":
        # no wrapping needed for kitchen
        args.domain = "kitchen"
        detector = KitchenDetector(wrapped_env)
        plan = load_plan(config['planning'][args.domain])
        grounded_operator = find_grounded_operator_from_plan(
            plan, 'place-on-object')
        # load the executor
        from execution.kitchen.kitchen_executor import KITCHEN_EXECUTORS
        executor = KITCHEN_EXECUTORS['place-on-object']
        success = executor.execute(detector, grounded_operator, render=True)
        if "Pre_Novelty" not in options["env_name"]:
            wrapped_env = choose_op_wrap("kitchen", wrapped_env)
            wrapped_env = choose_subgoal(wrapped_env)
            wrapped_env = choose_reward_shaping_fn(wrapped_env)
    else:
        raise ValueError("Unrecognized environment name: {}".format(options["env_name"]))

    # keyboard control
    from robosuite.devices import Keyboard

    device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
    wrapped_env.viewer.add_keypress_callback(device.on_press)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    

    while True:
        # Reset the environment
        obs = wrapped_env.reset()
        camera_id = 2
        print(env.sim.model.camera_names)
        print(env.sim.model.camera_names[camera_id])
        env.viewer.set_camera(camera_id=camera_id)
        wrapped_env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = wrapped_env.robots[0]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm='left', env_configuration="single-arm-opposed"
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                # Update last grasp
                last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = wrapped_env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                
                action = np.concatenate([rem_action, action])
                
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: wrapped_env.action_dim]

            # Step through the simulation and render
            obs, reward, done, *_ = wrapped_env.step(action)
            if detector is not None:
                groundings = detector.detect_binary_states(env)
                obs = detector.get_obs()
                for i, effect in enumerate(wrapped_env.grounded_operator.effects):
                    if effect.atom.pddl_repr() in groundings:
                        print(effect.atom.pddl_repr(), ":", groundings[effect.atom.pddl_repr()])
                print('action: ', action)
                print('reward: ', reward)
                print(obs)
            wrapped_env.render()

