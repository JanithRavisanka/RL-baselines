import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latest_model_path(results_subdir: str, filename: str) -> Path:
    base = ROOT / "results" / results_subdir
    if not base.exists():
        raise FileNotFoundError(f"Results folder does not exist: {base}")
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No '{filename}' found under {base}/run_*")


def list_model_paths(results_subdir: str, filename: str):
    base = ROOT / "results" / results_subdir
    if not base.exists():
        return []
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
    models = []
    for run_dir in run_dirs:
        candidate = run_dir / filename
        if candidate.exists():
            models.append(candidate)
    return models


def choose_from_menu(prompt: str, options: list) -> str:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid choice. Enter a valid number.")


def interactive_select_algo_and_model() -> Tuple[str, Optional[Path]]:
    algo_choices = [
        "actor_critic",
        "actor_critic_continuous",
        "dqn",
        "ddqn",
        "per_ddqn",
        "dyna_q",
        "mpc",
        "muzero",
    ]
    print("Available algorithms:")
    for i, algo in enumerate(algo_choices, start=1):
        print(f"  {i}. {algo}")
    selected_algo = choose_from_menu("Choose algorithm number: ", algo_choices)

    results_subdir, filename = model_file_for_algo(selected_algo)
    models = list_model_paths(results_subdir, filename)
    if not models:
        print(f"No models found for '{selected_algo}' in results/{results_subdir}/run_*/{filename}")
        return selected_algo, None

    print(f"Available models for '{selected_algo}':")
    for i, path in enumerate(models, start=1):
        print(f"  {i}. {path}")
    print("Press Enter to use latest model.")
    selected = input("Choose model number (default latest): ").strip()
    if selected == "":
        return selected_algo, models[0]
    if selected.isdigit():
        idx = int(selected) - 1
        if 0 <= idx < len(models):
            return selected_algo, models[idx]
    print("Invalid model choice. Using latest model.")
    return selected_algo, models[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play trained RL checkpoints across all algorithms.")
    parser.add_argument(
        "--algo",
        required=False,
        choices=[
            "actor_critic",
            "actor_critic_continuous",
            "dqn",
            "ddqn",
            "per_ddqn",
            "dyna_q",
            "mpc",
            "muzero",
        ],
        help="Algorithm key to evaluate.",
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play.")
    parser.add_argument("--render", choices=["human", "rgb_array"], default="human", help="Render mode.")
    parser.add_argument("--save-gif", default=None, help="Optional GIF output path.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode safety cap.")
    return parser


def model_file_for_algo(algo: str) -> Tuple[str, str]:
    mapping = {
        "actor_critic": ("actor_critic", "model.pth"),
        "actor_critic_continuous": ("actor_critic_continuous", "model.pth"),
        "dqn": ("dqn", "model.pth"),
        "ddqn": ("ddqn", "model.pth"),
        "per_ddqn": ("per_ddqn", "model.pth"),
        "dyna_q": ("dyna_q", "q_table.npy"),
        "mpc": ("mpc", "dynamics_model.pth"),
        "muzero": ("muzero", "muzero_network.pth"),
    }
    return mapping[algo]


def load_runtime(algo: str, checkpoint: Path, device: torch.device, render_mode: str):
    if algo == "actor_critic":
        module = load_module("actor_critic_mod", ROOT / "baselines" / "model-free" / "actor-critic" / "actor_critic.py")
        env = gym.make("CartPole-v1", render_mode=render_mode)
        model = module.ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        return {"env": env, "module": module, "model": model}

    if algo == "actor_critic_continuous":
        module = load_module(
            "actor_critic_cont_mod",
            ROOT / "baselines" / "model-free" / "actor-critic" / "actor_critic_continuous.py",
        )
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        model = module.ActorCriticContinuous(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        return {"env": env, "module": module, "model": model}

    if algo == "dqn":
        module = load_module("dqn_mod", ROOT / "baselines" / "model-free" / "DQN" / "dqn.py")
        env = module.make_env(render_mode=render_mode)
        model = module.QNetwork(env.action_space.n).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        return {"env": env, "module": module, "model": model, "requires_fire_start": True}

    if algo == "ddqn":
        module = load_module("ddqn_mod", ROOT / "baselines" / "model-free" / "DDQN" / "double_dqn.py")
        env = module.make_env(render_mode=render_mode)
        model = module.QNetwork(env.action_space.n).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        return {"env": env, "module": module, "model": model, "requires_fire_start": True}

    if algo == "per_ddqn":
        module = load_module("per_mod", ROOT / "baselines" / "model-free" / "PER" / "per_ddqn.py")
        env = module.make_env(render_mode=render_mode)
        model = module.QNetwork(env.action_space.n).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        return {"env": env, "module": module, "model": model, "requires_fire_start": True}

    if algo == "dyna_q":
        env = gym.make("CliffWalking-v0", render_mode=render_mode)
        q_table = np.load(checkpoint)
        return {"env": env, "q_table": q_table}

    if algo == "mpc":
        module = load_module("mpc_mod", ROOT / "baselines" / "model-based" / "MPC" / "learned_dynamics_mpc.py")
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        model = module.DynamicsModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        planner = module.RandomShootingPlanner(model, env.action_space.shape[0], num_sequences=2000, horizon=15)
        return {"env": env, "planner": planner}

    if algo == "muzero":
        module = load_module("muzero_mod", ROOT / "baselines" / "model-based" / "MuZero" / "muzero.py")
        env = gym.make("CartPole-v1", render_mode=render_mode)
        network = module.MuZeroNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
        network.load_state_dict(torch.load(checkpoint, map_location=device))
        network.eval()
        config = {
            "action_dim": env.action_space.n,
            "num_simulations": 25,
            "discount": 0.99,
            "pb_c_init": 1.25,
            "dirichlet_alpha": 0.25,
            "dirichlet_eps": 0.0,
        }
        return {"env": env, "module": module, "network": network, "config": config}

    raise ValueError(f"Unsupported algorithm: {algo}")


def select_action(algo: str, state: np.ndarray, runtime: Dict[str, Any], device: torch.device) -> Any:
    with torch.no_grad():
        if algo == "actor_critic":
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = runtime["model"](state_tensor)
            return torch.argmax(action_probs, dim=1).item()

        if algo == "actor_critic_continuous":
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mu, _, _ = runtime["model"](state_tensor)
            action = mu.cpu().numpy()[0]
            return np.clip(action, -2.0, 2.0)

        if algo in {"dqn", "ddqn", "per_ddqn"}:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0
            q_values = runtime["model"](state_tensor)
            return torch.argmax(q_values, dim=1).item()

        if algo == "dyna_q":
            q_table = runtime["q_table"]
            return int(np.argmax(q_table[int(state)]))

        if algo == "mpc":
            return runtime["planner"].get_action(state)

        if algo == "muzero":
            policy, _ = runtime["module"].run_mctx(runtime["config"], runtime["network"], state)
            return int(np.argmax(policy))

    raise ValueError(f"Unsupported algorithm: {algo}")


def maybe_capture_frame(env, frames: list, render_mode: str):
    if render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            frames.append(frame)


def run_episode(algo: str, runtime: Dict[str, Any], device: torch.device, max_steps: int, render_mode: str):
    env = runtime["env"]
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []

    if runtime.get("requires_fire_start", False):
        maybe_capture_frame(env, frames, render_mode)
        state, reward, terminated, truncated, _ = env.step(1)
        total_reward += reward
        done = terminated or truncated

    steps = 0
    while not done and steps < max_steps:
        maybe_capture_frame(env, frames, render_mode)
        action = select_action(algo, state, runtime, device)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    maybe_capture_frame(env, frames, render_mode)
    return total_reward, steps, frames


def main():
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)

    if args.save_gif and args.render == "human":
        print("Warning: --save-gif requested with --render human. Switching render mode to rgb_array.")
        render_mode = "rgb_array"
    else:
        render_mode = args.render

    if not args.algo:
        selected_algo, selected_model = interactive_select_algo_and_model()
        args.algo = selected_algo
        if selected_model is not None:
            args.model_path = str(selected_model)

    if args.model_path:
        checkpoint = Path(args.model_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Model path does not exist: {checkpoint}")
    else:
        results_subdir, filename = model_file_for_algo(args.algo)
        checkpoint = latest_model_path(results_subdir, filename)

    print(f"Algorithm: {args.algo}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    runtime = load_runtime(args.algo, checkpoint, device, render_mode)
    rewards = []
    all_frames = []
    for ep in range(1, args.episodes + 1):
        total_reward, steps, frames = run_episode(args.algo, runtime, device, args.max_steps, render_mode)
        rewards.append(total_reward)
        if args.save_gif:
            all_frames.extend(frames)
        print(f"Episode {ep}: reward={total_reward:.2f}, steps={steps}")

    runtime["env"].close()
    print(f"Average reward over {len(rewards)} episode(s): {np.mean(rewards):.2f}")

    if args.save_gif:
        if not all_frames:
            raise RuntimeError("No frames were captured. Use rgb_array render mode for GIF export.")
        gif_path = Path(args.save_gif).expanduser().resolve()
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, all_frames, fps=30)
        print(f"Saved GIF to: {gif_path}")


if __name__ == "__main__":
    main()
