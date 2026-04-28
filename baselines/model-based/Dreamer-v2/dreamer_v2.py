import argparse
from collections import deque
from pathlib import Path
import sys

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dreamer_common import ConvDecoder, ConvEncoder, DiscreteRSSM, DreamerConfig, MLPHead, lambda_return


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity=400_000):
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
            torch.tensor(np.array(act), dtype=torch.long),
            torch.tensor(np.array(rew), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )


class DreamerV2(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(3, depth=48)
        with torch.no_grad():
            enc_dim = self.encoder(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.rssm = DiscreteRSSM(action_dim, deter=600, stoch=32, classes=32, hidden=600, embed_dim=enc_dim)
        latent_dim = 600 + 32 * 32
        self.decoder = ConvDecoder(out_channels=3, depth=48, emb=latent_dim)
        self.reward = MLPHead(latent_dim, 1, hidden=600, layers=2)
        self.value = MLPHead(latent_dim, 1, hidden=600, layers=3)
        self.actor = MLPHead(latent_dim, action_dim, hidden=600, layers=4)

    def feat(self, state):
        return torch.cat([state["deter"], state["stoch"].flatten(start_dim=1)], dim=-1)

    def actor_dist(self, feat):
        logits = self.actor(feat)
        return torch.distributions.Categorical(logits=logits)


def preprocess_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame = F.interpolate(frame, size=(64, 64), mode="bilinear", align_corners=False)
    return frame.squeeze(0).byte().numpy()


def train(args):
    cfg = DreamerConfig(
        horizon=15,
        gamma=0.997,
        lambda_=0.95,
        world_lr=3e-4,
        actor_lr=1e-4,
        value_lr=1e-4,
        kl_scale=1.0,
        free_nats=1.0,
        batch_size=16,
        seq_len=50,
        latent_dim=32 * 32,
        deter_dim=600,
        atari=True,
    )
    dev = device()
    env = gym.make(args.env, render_mode="rgb_array")
    action_dim = env.action_space.n
    model = DreamerV2(action_dim).to(dev)

    world_params = list(model.encoder.parameters()) + list(model.rssm.parameters()) + list(model.decoder.parameters()) + list(model.reward.parameters())
    world_opt = optim.Adam(world_params, lr=cfg.world_lr, eps=1e-5)
    actor_opt = optim.Adam(model.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
    value_opt = optim.Adam(model.value.parameters(), lr=cfg.value_lr, eps=1e-5)
    replay = ReplayBuffer()

    obs, _ = env.reset()
    obs = preprocess_frame(obs)
    for step in range(args.prefill):
        action = env.action_space.sample()
        nxt, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        replay.add(obs, action, rew, float(done))
        obs = preprocess_frame(nxt if not done else env.reset()[0])
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
            action_oh = F.one_hot(act_b[:, t], action_dim).float()
            embed = model.encoder(obs_b[:, t])
            post, prior = model.rssm.observe_step(state, action_oh, embed)
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
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 3e-4 * entropy).mean()

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

        if update % 100 == 0:
            print(
                f"Update {update}/{args.updates} | world {world_loss.item():.4f} "
                f"| actor {actor_loss.item():.4f} | value {value_loss.item():.4f}"
            )


def build_args():
    p = argparse.ArgumentParser(description="Dreamer V2 (paper architecture) in PyTorch")
    p.add_argument("--env", type=str, default="ALE/Breakout-v5")
    p.add_argument("--prefill", type=int, default=20000)
    p.add_argument("--updates", type=int, default=3000)
    return p.parse_args()


if __name__ == "__main__":
    train(build_args())
