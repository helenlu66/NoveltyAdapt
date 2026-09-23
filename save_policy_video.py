import argparse
import importlib
import os
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor

from learning.custom_gym_wrapper import CollisionAblatedOperatorWrapper, CollisionLLMAblatedOperatorWrapper
from learning.learning_utils import filter_observation, get_observation_with_semantics, choose_grounded_op
from utils import extract_name_params_from_grounded, load_config, load_env, load_plan


def resolve_grounded_operator(config: dict, domain: str, op_name: Optional[str], op_index: Optional[int]):
	"""Resolve grounded operator from args or fallback to interactive chooser."""
	plan = load_plan(config["planning"][domain])
	if plan is None or len(plan) == 0:
		raise ValueError(f"No grounded operators found in planning config for domain '{domain}'.")

	if op_index is not None:
		if op_index < 0 or op_index >= len(plan):
			raise ValueError(f"--grounded_op_idx must be in [0, {len(plan) - 1}], got {op_index}.")
		return plan[op_index]

	if op_name:
		for op in plan:
			if op.name == op_name or op_name in op.name:
				return op
		valid_names = [op.name for op in plan]
		raise ValueError(f"Grounded operator '{op_name}' not found. Available operators: {valid_names}")

	# Keep parity with training utilities when no selector is given.
	return choose_grounded_op(config, domain)


def infer_grounded_operator_from_model_path(config: dict, domain: str, model_path: str):
	"""Infer grounded operator from policy path segment to avoid interactive prompts."""
	plan = load_plan(config["planning"][domain])
	if plan is None or len(plan) == 0:
		raise ValueError(f"No grounded operators found in planning config for domain '{domain}'.")

	normalized = model_path.replace("\\", "/")
	parts = [p for p in normalized.split("/") if p]

	# Path shape: .../learning/policies/<domain>/<agent>/<op_name>/<algo>/...
	for i, token in enumerate(parts):
		if token in {"llm_subgoal", "subgoal"} and i + 1 < len(parts):
			op_name_from_path = parts[i + 1]
			for op in plan:
				if op.name == op_name_from_path:
					return op
			for op in plan:
				if op_name_from_path in op.name or op.name in op_name_from_path:
					return op

	valid_names = [op.name for op in plan]
	raise ValueError(
		"Could not infer grounded operator from --model_path. "
		f"Please pass --grounded_op or --grounded_op_idx. Available operators: {valid_names}"
	)


def resolve_subgoal(grounded_op):
	"""Always use the last effect in grounded operator as the target subgoal."""
	if len(grounded_op.effects) == 0:
		raise ValueError(f"Operator {grounded_op.name} has no effects/subgoals.")
	return grounded_op.effects[-1]


def build_candidate_model_paths(args, op_name: str) -> List[str]:
	"""Generate likely operator-level checkpoint paths (final_model.zip)."""
	base = os.path.join("learning", "policies", args.domain, args.agent, op_name, args.rl_algorithm)

	candidates: List[str] = [
		# Current learner layout.
		os.path.join(base, args.termination_type, f"seed_{args.seed}_{args.rw_type}", "final_model.zip"),
		os.path.join(base, args.termination_type, f"seed_{args.seed}_{args.rw_type}", "final_model"),
		# Common fallback if rw_type segment is absent.
		os.path.join(base, args.termination_type, f"seed_{args.seed}", "final_model.zip"),
		os.path.join(base, args.termination_type, f"seed_{args.seed}", "final_model"),
		# Older layouts.
		os.path.join(base, f"seed_{args.seed}", "final_model.zip"),
		os.path.join(base, f"seed_{args.seed}", "final_model"),
		os.path.join(base, f"seed_{args.seed}_replay_buffer", "final_model.zip"),
		os.path.join(base, f"seed_{args.seed}_replay_buffer", "final_model"),
	]

	return candidates


def get_model_path(args, op_name: str) -> str:
	"""Resolve operator-level model path from explicit input or known conventions."""
	if args.model_path:
		if os.path.exists(args.model_path):
			return args.model_path
		raise FileNotFoundError(f"Provided --model_path does not exist: {args.model_path}")

	for candidate in build_candidate_model_paths(args, op_name):
		if os.path.exists(candidate):
			return candidate

	attempted = "\n".join(build_candidate_model_paths(args, op_name))
	raise FileNotFoundError(
		"Could not find a policy checkpoint. Attempted paths:\n"
		f"{attempted}\n"
		"You can pass --model_path explicitly."
	)


def resolve_camera_name_from_obs(obs_dict: dict, requested_camera_name: Optional[str]) -> str:
	"""Resolve camera name from observation keys, with fallback to first available image key."""
	image_keys = [k for k in obs_dict.keys() if k.endswith("_image")]
	if len(image_keys) == 0:
		raise KeyError("No *_image keys found in observation. Camera observations are unavailable.")

	if requested_camera_name is not None:
		requested_key = f"{requested_camera_name}_image"
		if requested_key in obs_dict:
			return requested_camera_name

	return image_keys[0][: -len("_image")]


def frame_from_observation(obs_dict: dict, camera_name: str) -> np.ndarray:
	"""Extract RGB frame from observation dict produced by robosuite camera sensors."""
	key = f"{camera_name}_image"
	if key not in obs_dict:
		available_image_keys = [k for k in obs_dict.keys() if k.endswith("_image")]
		raise KeyError(f"Camera key '{key}' not found in observation. Available image keys: {available_image_keys}")

	frame = obs_dict[key]
	if frame.dtype != np.uint8:
		frame = np.clip(frame, 0, 255).astype(np.uint8)
	if frame.ndim == 3 and frame.shape[-1] == 4:
		frame = frame[..., :3]
	# Align saved video orientation with expected on-screen view.
	frame = np.flipud(frame)
	return frame


def resolve_output_path(output_arg: str, op_name: str, seed: int) -> str:
	"""Build output path that includes operator name unless an explicit mp4 filename is provided."""
	op_stem = op_name.replace("(", "").replace(")", "").replace(" ", "_")
	default_name = f"{op_stem}_seed_{seed}.mp4"

	if output_arg is None or output_arg == "":
		return os.path.join("outputs", default_name)

	# If user passed a directory-like path, create a named file under it.
	if output_arg.endswith(os.sep) or (os.path.exists(output_arg) and os.path.isdir(output_arg)):
		return os.path.join(output_arg, default_name)

	# If no extension was provided, treat as directory-like stem.
	root, ext = os.path.splitext(output_arg)
	if ext == "":
		return os.path.join(output_arg, default_name)

	# Explicit file path provided; keep it.
	return output_arg


def main():
	config = load_config("config.yaml")

	parser = argparse.ArgumentParser(description="Save a policy rollout video.")
	parser.add_argument("--domain", choices=["nut_assembly", "kitchen", "coffee_drawer", "coffee_box"], required=True)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--agent", choices=["llm_subgoal", "subgoal"], default="llm_subgoal")
	parser.add_argument("--rl_algorithm", type=str, default="PPO")
	parser.add_argument("--ep_len", type=int, default=75)
	parser.add_argument(
		"--termination_type",
		choices=["on_first_subgoal_success", "on_all_subgoal_success", "no_termination", "on_non_terminal_subgoal_success"],
		default="on_first_subgoal_success",
	)
	parser.add_argument(
		"--rw_type",
		choices=[
			"pos_linear_dist",
			"pos_linear_vel",
			"pos_exp_dist",
			"pos_exp_dist_test",
			"pos_exp_vel",
			"pos_linear_vel_test",
			"pos_linear_vel_test_range",
			"neg_exp_dist",
			"neg_exp_vel",
		],
		default="pos_linear_vel",
	)
	parser.add_argument("--rw_shaping", type=int, default=0, help="Reward fn index for llm_subgoal agent.")
	parser.add_argument("--replay_buffer", action="store_true")
	parser.add_argument("--grounded_op", type=str, default=None, help="Grounded operator name (exact or substring).")
	parser.add_argument("--grounded_op_idx", type=int, default=None, help="Index of grounded operator in the plan.")
	parser.add_argument("--model_path", type=str, default=None, help="Explicit path to model checkpoint (.zip or SB3 path stem).")
	parser.add_argument("--output", type=str, default=None, help="Output file path or output directory.")
	parser.add_argument("--fps", type=int, default=20)
	parser.add_argument("--width", type=int, default=640)
	parser.add_argument("--height", type=int, default=480)
	parser.add_argument("--camera_name", type=str, default=None)
	parser.add_argument("--camera_id", type=int, default=0)
	parser.add_argument("--deterministic", action="store_true")
	parser.set_defaults(deterministic=True)
	args = parser.parse_args()

	np.random.seed(args.seed)

	# Enable offscreen rendering for video capture.
	config["simulation"]["has_renderer"] = False
	config["simulation"]["has_offscreen_renderer"] = True
	config["simulation"]["use_camera_obs"] = True
	config["simulation"]["camera_heights"] = args.height
	config["simulation"]["camera_widths"] = args.width
	config["simulation"]["camera_names"] = args.camera_name if args.camera_name is not None else "agentview"

	robosuite_env = load_env(args.domain, config["simulation"])

	model_path = None
	if args.grounded_op is None and args.grounded_op_idx is None and args.model_path:
		grounded_op = infer_grounded_operator_from_model_path(config=config, domain=args.domain, model_path=args.model_path)
		subgoal = resolve_subgoal(grounded_op=grounded_op)
	else:
		grounded_op = resolve_grounded_operator(config=config, domain=args.domain, op_name=args.grounded_op, op_index=args.grounded_op_idx)
		subgoal = resolve_subgoal(grounded_op=grounded_op)

	filtered_obs = filter_observation(get_observation_with_semantics(robosuite_env), grounded_op)
	gym_env = GymWrapper(robosuite_env, keys=list(filtered_obs.keys()))
	gym_env = TimeLimit(gym_env, max_episode_steps=args.ep_len)

	if args.agent == "llm_subgoal":
		wrapped_env = CollisionAblatedOperatorWrapper(
			env=gym_env,
			grounded_operator=grounded_op,
			config=config,
			args=args,
			curr_subgoal=subgoal,
		)
	else:
		wrapped_env = CollisionLLMAblatedOperatorWrapper(
			env=gym_env,
			grounded_operator=grounded_op,
			config=config,
			args=args,
			curr_subgoal=subgoal,
		)

	env = Monitor(wrapped_env, filename=f"{args.domain}_video_monitor", allow_early_resets=True)

	rl_algo = importlib.import_module(f"stable_baselines3.{args.rl_algorithm.lower()}").__dict__[args.rl_algorithm.upper()]

	if model_path is None:
		op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
		op_name = op_name.replace("(", "_").replace(")", "_").replace(" ", "_")
		model_path = get_model_path(args=args, op_name=op_name)
	else:
		op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
	print(f"model_path: {model_path}")
	print(f"Loading model from: {model_path}")

	model = rl_algo.load(model_path, env=env)
	output_path = resolve_output_path(args.output, op_name=op_name, seed=args.seed)

	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	obs, info = env.reset(seed=args.seed)
	obs_dict = robosuite_env._get_observations()
	resolved_camera_name = resolve_camera_name_from_obs(obs_dict=obs_dict, requested_camera_name=args.camera_name)
	print(f"Using camera '{resolved_camera_name}' for video capture")
	writer = imageio.get_writer(output_path, fps=args.fps)

	for _ in range(args.ep_len):
		action, _ = model.predict(obs, deterministic=args.deterministic)
		obs, reward, done, truncated, info = env.step(action)
		obs_dict = robosuite_env._get_observations()
		frame = frame_from_observation(obs_dict=obs_dict, camera_name=resolved_camera_name)
		writer.append_data(frame)

		if done or truncated:
			obs, info = env.reset()

	writer.close()
	env.close()
	print(f"Saved rollout video to: {output_path}")


if __name__ == "__main__":
	main()
