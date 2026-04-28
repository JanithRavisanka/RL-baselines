import argparse
from collections import deque
from pathlib import Path
import sys
import os
import datetime

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import shimmy
gym.register_envs(shimmy)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dreamer_common import (
    ConvDecoder,
    ConvEncoder,
    ContinuousRSSM,
    DreamerConfig,
    MLPHead,
    lambda_return,
)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.obs = deque(maxlen=capacity)
        self.act = deque(maxlen=capacity)
        self.rew = deque(maxlen=capacity)
        self.done = deque(maxlen=capacity)

    def add(self, o, a, r, d):
        self.obs.append(o)
        self.act.append(a)
        self.rew.append(r)
        self.done.append(d)

    def __len__(self):
        return len(self.obs)

    def sample(self, batch, seq):
        max_i = len(self.obs) - seq - 1
        idx = np.random.randint(0, max_i, size=batch)
        obs, act, rew, done = [], [], [], []
        for i in idx:
            obs.append(np.stack([self.obs[i + t] for t in range(seq + 1)], axis=0))
            act.append(np.stack([self.act[i + t] for t in range(seq)], axis=0))
            rew.append(np.stack([self.rew[i + t] for t in range(seq)], axis=0))
            done.append(np.stack([self.done[i + t] for t in range(seq)], axis=0))
        return (
            torch.tensor(np.array(obs), dtype=torch.uint8),
            torch.tensor(np.array(act), dtype=torch.float32),
            torch.tensor(np.array(rew), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )


class DreamerV1(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(3, depth=32)
        with torch.no_grad():
            enc_dim = self.encoder(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.rssm = ContinuousRSSM(action_dim, deter=200, stoch=30, hidden=200, embed_dim=enc_dim)
        latent_dim = 200 + 30
        self.decoder = ConvDecoder(out_channels=3, depth=32, emb=latent_dim)
        self.reward = MLPHead(latent_dim, 1, hidden=200, layers=2)
        self.value = MLPHead(latent_dim, 1, hidden=200, layers=3)
        self.actor = MLPHead(latent_dim, action_dim * 2, hidden=200, layers=4)

    def feat(self, state):
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def actor_dist(self, feat):
        mean, std_logits = torch.chunk(self.actor(feat), 2, dim=-1)
        std = F.softplus(std_logits + 0.54) + 0.1
        return torch.distributions.Normal(torch.tanh(mean), std)


def preprocess_frame(frame):
    frame = np.ascontiguousarray(np.asarray(frame))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame = F.interpolate(frame, size=(64, 64), mode="bilinear", align_corners=False)
    return frame.squeeze(0).byte().numpy()


def to_pixel_observation(env, obs):
    arr = np.asarray(obs) if not isinstance(obs, dict) else np.array([], dtype=np.float32)
    if arr.dtype != np.object_ and arr.ndim in (2, 3):
        return preprocess_frame(arr)
    # dm_control via shimmy typically returns dict observations; use rendered pixels.
    rendered = env.render()
    if rendered is None:
        raise RuntimeError("Environment did not return pixel frames; cannot run Dreamer V1 pixel pipeline.")
    return preprocess_frame(rendered)


def train(args):
    cfg = DreamerConfig(
        horizon=15,
        gamma=0.99,
        lambda_=0.95,
        world_lr=6e-4,
        actor_lr=8e-5,
        value_lr=8e-5,
        kl_scale=1.0,
        free_nats=3.0,
        batch_size=32,
        seq_len=50,
        latent_dim=30,
        deter_dim=200,
    )
    dev = device()
    env = gym.make(args.env, render_mode="rgb_array")
    action_dim = env.action_space.shape[0]
    model = DreamerV1(action_dim).to(dev)
    world_params = list(model.encoder.parameters()) + list(model.rssm.parameters()) + list(model.decoder.parameters()) + list(model.reward.parameters())
    world_opt = optim.Adam(world_params, lr=cfg.world_lr)
    actor_opt = optim.Adam(model.actor.parameters(), lr=cfg.actor_lr)
    value_opt = optim.Adam(model.value.parameters(), lr=cfg.value_lr)
    replay = ReplayBuffer()
    world_losses, actor_losses, value_losses = [], [], []

    obs, _ = env.reset()
    obs = to_pixel_observation(env, obs)
    for step in range(args.prefill):
        action = env.action_space.sample()
        nxt, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        replay.add(obs, action, rew, float(done))
        if done:
            nxt, _ = env.reset()
        obs = to_pixel_observation(env, nxt)
        if step % 1000 == 0 and step > 0:
            print(f"Prefill: {step}/{args.prefill}")

    for update in range(args.updates):
        obs_b, act_b, rew_b, done_b = replay.sample(cfg.batch_size, cfg.seq_len)
        obs_b = obs_b.to(dev)
        act_b = act_b.to(dev)
        rew_b = rew_b.to(dev)
        done_b = done_b.to(dev)

        state = model.rssm.init_state(cfg.batch_size, dev)
        posts, priors, feats = [], [], []
        for t in range(cfg.seq_len):
            embed = model.encoder(obs_b[:, t])
            post, prior = model.rssm.observe_step(state, act_b[:, t], embed)
            feat = model.feat(post)
            posts.append(post)
            priors.append(prior)
            feats.append(feat)
            state = post

        feats = torch.stack(feats, 0)
        recon = model.decoder(feats.reshape(-1, feats.shape[-1]))
        target = obs_b[:, 1:].transpose(0, 1).reshape(-1, 3, 64, 64).float() / 255.0
        recon_loss = F.mse_loss(torch.sigmoid(recon), target)
        reward_pred = model.reward(feats).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rew_b.transpose(0, 1))
        kl = torch.stack([model.rssm.kl(p, r) for p, r in zip(posts, priors)], 0)
        kl_loss = torch.clamp(kl.mean(), min=cfg.free_nats)
        world_loss = recon_loss + reward_loss + cfg.kl_scale * kl_loss

        world_opt.zero_grad()
        world_loss.backward()
        nn.utils.clip_grad_norm_(world_params, 100.0)
        world_opt.step()

        with torch.no_grad():
            value = model.value(feats).squeeze(-1)
            disc = (1.0 - done_b.transpose(0, 1)) * cfg.gamma
            target = lambda_return(rew_b.transpose(0, 1), value, disc, value[-1], cfg.lambda_)
            advantage = target - value

        dist = model.actor_dist(feats.detach())
        action = torch.clamp(dist.rsample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        actor_loss = -(log_prob * advantage.detach() + 1e-3 * entropy).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 100.0)
        actor_opt.step()

        value_pred = model.value(feats.detach()).squeeze(-1)
        value_loss = F.mse_loss(value_pred, target.detach())
        value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(model.value.parameters(), 100.0)
        value_opt.step()
        world_losses.append(float(world_loss.item()))
        actor_losses.append(float(actor_loss.item()))
        value_losses.append(float(value_loss.item()))

        if update % 100 == 0:
            print(
                f"Update {update}/{args.updates} | world {world_loss.item():.4f} "
                f"| actor {actor_loss.item():.4f} | value {value_loss.item():.4f}"
            )
    env.close()
    return model, {
        "world_loss": world_losses,
        "actor_loss": actor_losses,
        "value_loss": value_losses,
    }


def plot_metrics(metrics, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["world_loss"], label="world_loss")
    plt.plot(metrics["actor_loss"], label="actor_loss")
    plt.plot(metrics["value_loss"], label="value_loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Dreamer V1 Training Losses")
    plt.legend()
    plt.grid()
    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out)
    print(f"Training curve saved: {out}")


def evaluate_and_record(model, env_name, save_dir, dev, max_steps=1000):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    obs = to_pixel_observation(env, obs)
    state = model.rssm.init_state(batch=1, device=dev)
    prev_action = torch.zeros(1, env.action_space.shape[0], device=dev)
    frames = []
    total_reward = 0.0
    done = False
    steps = 0

    with torch.no_grad():
        while not done and steps < max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=dev).unsqueeze(0)
            embed = model.encoder(obs_t)
            post, _ = model.rssm.observe_step(state, prev_action, embed)
            feat = model.feat(post)
            dist = model.actor_dist(feat)
            action = torch.clamp(dist.mean, -1.0, 1.0)
            nxt, rew, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            done = term or trunc
            total_reward += rew
            obs = to_pixel_observation(env, nxt if not done else env.reset()[0])
            state = post
            prev_action = action
            steps += 1

    env.close()
    gif_path = os.path.join(save_dir, "dreamer_v1_agent.gif")
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Evaluation GIF saved: {gif_path}")
    print(f"Evaluation reward: {total_reward:.2f}")


def build_args():
    p = argparse.ArgumentParser(description="Dreamer V1 (paper architecture) in PyTorch")
    p.add_argument("--env", type=str, default="dm_control/walker-walk-v0")
    p.add_argument("--prefill", type=int, default=5000)
    p.add_argument("--updates", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    dev = device()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "dreamer_v1", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all outputs to: {save_dir}")

    model, metrics = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    plot_metrics(metrics, save_dir)
    evaluate_and_record(model, args.env, save_dir, dev)
