import csv
import datetime
import json
import os
import random

import numpy as np
import torch


EVAL_FIELDS = [
    "global_step",
    "eval_reward_mean",
    "eval_reward_std",
    "eval_reward_min",
    "eval_reward_max",
    "eval_steps_mean",
    "eval_episodes",
    "gif_path",
]


EPISODE_FIELDS = [
    "episode",
    "global_step",
    "reward",
    "episode_steps",
    "epsilon",
    "completed",
    "elapsed_sec",
]


TRAINING_FIELDS = [
    "global_step",
    "elapsed_sec",
    "episodes",
    "epsilon",
    "avg_reward_20",
    "avg_reward_100",
]


def seed_everything(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.action_space.seed(seed)


def namespace_to_dict(args):
    if args is None:
        return {}
    return vars(args).copy()


def write_config(args, save_dir, algorithm, device):
    path = os.path.join(save_dir, "config.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "algorithm": algorithm,
                "config": namespace_to_dict(args),
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "device": str(device),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return path


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def std_or_zero(values):
    return float(np.std(values)) if values else 0.0


def ensure_csv(path, fieldnames):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path, fieldnames, row):
    ensure_csv(path, fieldnames)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def fire_action_index(env):
    try:
        meanings = env.unwrapped.get_action_meanings()
    except AttributeError:
        return None
    return meanings.index("FIRE") if "FIRE" in meanings else None


def maybe_fire(env, state, done, total_reward, frames=None):
    fire_action = fire_action_index(env)
    if fire_action is None or done:
        return state, done, total_reward
    state, reward, terminated, truncated, _ = env.step(fire_action)
    done = terminated or truncated
    total_reward += reward
    if frames is not None:
        frames.append(env.render())
    return state, done, total_reward


def evaluate_q_policy(
    model,
    make_env,
    device,
    env_name,
    global_step,
    eval_episodes,
    eval_max_steps,
    record_gif=False,
    gif_path=None,
):
    if record_gif:
        import imageio
    else:
        imageio = None

    rewards = []
    steps_per_episode = []
    frames = [] if record_gif else None
    was_training = model.training
    model.eval()

    env = make_env(env_name=env_name, render_mode="rgb_array" if record_gif else None, terminal_on_life_loss=False)
    try:
        with torch.no_grad():
            for episode in range(eval_episodes):
                state, _ = env.reset()
                done = False
                total_reward = 0.0
                steps = 0
                state, done, total_reward = maybe_fire(
                    env,
                    state,
                    done,
                    total_reward,
                    frames if episode == 0 else None,
                )
                steps += 1
                lives = env.unwrapped.ale.lives() if hasattr(env.unwrapped, "ale") else None

                while not done and steps < eval_max_steps:
                    if frames is not None and episode == 0:
                        frames.append(env.render())

                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                    action = int(model(state_tensor).argmax().item())
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1

                    if not done:
                        current_lives = env.unwrapped.ale.lives() if hasattr(env.unwrapped, "ale") else None
                        if lives is not None and current_lives is not None and current_lives < lives:
                            lives = current_lives
                            state, done, total_reward = maybe_fire(
                                env,
                                state,
                                done,
                                total_reward,
                                frames if episode == 0 else None,
                            )
                            steps += 1
                        elif current_lives is not None:
                            lives = current_lives

                rewards.append(float(total_reward))
                steps_per_episode.append(int(steps))
    finally:
        env.close()
        if was_training:
            model.train()

    if record_gif and gif_path and frames:
        imageio.mimsave(gif_path, frames, fps=30)

    return {
        "global_step": int(global_step),
        "eval_reward_mean": mean_or_zero(rewards),
        "eval_reward_std": std_or_zero(rewards),
        "eval_reward_min": float(np.min(rewards)) if rewards else 0.0,
        "eval_reward_max": float(np.max(rewards)) if rewards else 0.0,
        "eval_steps_mean": mean_or_zero(steps_per_episode),
        "eval_episodes": int(eval_episodes),
        "gif_path": "" if not gif_path else gif_path,
    }


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_final_summary(path, rewards, total_steps):
    write_json(
        path,
        {
            "episodes": len(rewards),
            "reward_mean": mean_or_zero(rewards),
            "reward_last_20_mean": mean_or_zero(rewards[-20:]),
            "reward_last_100_mean": mean_or_zero(rewards[-100:]),
            "reward_max": float(np.max(rewards)) if rewards else 0.0,
            "total_steps": int(total_steps),
        },
    )
