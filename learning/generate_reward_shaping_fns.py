import tarski.fstrips as fs
import argparse
from typing import *
from utils import *
from learning_utils import *
from learning.custom_gym_wrapper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt the LLM for reward shaping function candidates for a grounded operator')
    parser.add_argument('--domain', type=str, default='nut_assembly', help='the domain')
    parser.add_argument('--rw_type', choices=['pos_linear_dist', 'pos_linear_vel', 'pos_exp_dist', 'pos_exp_vel','neg_exp_dist', 'neg_exp_vel'], default='pos_linear_vel', help='the type of reward shaping function to use')
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=False)  # Default value
    args = parser.parse_args()
    config = load_config("config.yaml")
    
    grounded_op = choose_grounded_op(config, args.domain)
    op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
    env = load_env(domain=args.domain, config = config['simulation'])
    prev_executors = find_prev_operators_executors(config=config, domain=args.domain, grounded_op=grounded_op)
    detector = load_detector(config=config, domain=args.domain, env=env)
    obs = env.reset()
    # do nothing (all zeros) for a few timesteps
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    for op, executor in prev_executors.items():
        success = executor.execute(detector=detector, grounded_operator=op, render=args.render)
    for i in range(config['learning']['reward_shaping_fn']['num_candidates']):
        reward_shaping_fn_candidate = prompt_llm_for_reward_shaping_fn_candidate(grounded_op, obs, args.rw_type)
        # save the output python function to a file in the reward_functions directory
        # create the directory if it does not exist
        if not os.path.exists(f"learning{os.sep}reward_functions{os.sep}{args.domain}"):
            os.makedirs(f"learning{os.sep}reward_functions{os.sep}{args.domain}")
        # create a file with the operator's name and save the function in it
        op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
        with open(f"learning{os.sep}reward_functions{os.sep}{args.domain}{os.sep}{op_name}_{args.rw_type}_{i}.py", 'w') as f:
            f.write(reward_shaping_fn_candidate)
    