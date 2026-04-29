import argparse
from collections import deque
from pathlib import Path
import sys
import os
import datetime

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dreamer_common import ConvDecoder, ConvEncoder, DiscreteRSSM, MLPHead, free_nats_loss, lambda_return, symexp, symlog
# symlog/symexp are Dreamer-style signed transforms:
# - symlog compresses large-magnitude targets while keeping sign information.
# - symexp inverts that compression back to the original reward scale.
# This allows stable regression on heavy-tailed rewards without discarding
# directional information around zero (unlike plain log on positive values).


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
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


def unimix_probs(probs, mix=0.01):
    classes = probs.shape[-1]
    return (1 - mix) * probs + mix * (1.0 / classes)


class DreamerV3(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = ConvEncoder(3, depth=64)
        with torch.no_grad():
            enc_dim = self.encoder(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.rssm = DiscreteRSSM(action_dim, deter=1024, stoch=32, classes=32, hidden=1024, embed_dim=enc_dim)
        latent_dim = 1024 + 32 * 32
        self.decoder = ConvDecoder(out_channels=3, depth=64, emb=latent_dim)
        self.reward = MLPHead(latent_dim, 1, hidden=1024, layers=3)
        self.cont = MLPHead(latent_dim, 1, hidden=1024, layers=2)
        self.value = MLPHead(latent_dim, 1, hidden=1024, layers=4)
        self.actor = MLPHead(latent_dim, action_dim, hidden=1024, layers=4)
        self.action_dim = action_dim

    def feat(self, state):
        return torch.cat([state["deter"], state["stoch"].flatten(start_dim=1)], dim=-1)

    def actor_dist(self, feat):
        logits = self.actor(feat)
        return torch.distributions.Categorical(logits=logits)


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


def mask_state(state, done):
    # Replay samples can cross episode boundaries. World-model sequence training
    # is still done as fixed-length chunks for efficiency, but latent carry-over
    # across terminal transitions would leak information between unrelated episodes.
    # Masking sets latent state to zero for finished trajectories so each sample
    # continues independently after done=True even inside one tensorized batch.
    alive = 1.0 - done
    return {
        "deter": state["deter"] * alive.unsqueeze(-1),
        "stoch": state["stoch"] * alive.view(-1, 1, 1),
        "probs": state.get("probs", state["stoch"]) * alive.view(-1, 1, 1),
    }


def apply_unimix_state(state, mix=0.01):
    if "probs" not in state:
        return state
    return {**state, "probs": unimix_probs(state["probs"], mix=mix)}


def imagine_behavior(model, start, horizon, action_dim, gamma, lambda_):
    # Policy/value are trained on imagined rollouts in latent space instead of
    # expensive pixel-space environment interactions. The world model provides
    # transitions and reward/continuation predictions, then actor-critic updates
    # are computed from these synthetic trajectories.
    state = detach_state(start)
    imag_feats, rewards, values, discounts, logps, entropies = [], [], [], [], [], []
    for _ in range(horizon):
        feat = model.feat(state)
        dist = model.actor_dist(feat)
        action = dist.sample()
        probs = dist.probs
        action_oh = F.one_hot(action, action_dim).float()
        # Straight-through gradient estimator:
        # forward pass uses sampled one-hot action, backward pass receives
        # gradients through distribution probabilities.
        action_oh = action_oh + probs - probs.detach()
        state = apply_unimix_state(model.rssm.imagine_step(state, action_oh), mix=0.01)
        imag_feat = model.feat(state)
        # Continuation head predicts P(not terminal). Multiplying by gamma yields
        # per-step discount used by lambda-return for imagined trajectories.
        cont = torch.sigmoid(model.cont(imag_feat).squeeze(-1)) * gamma
        imag_feats.append(imag_feat)
        # Reward head is trained in symlog space for stability, but return targets
        # must be on the original reward scale; symexp performs that inversion.
        rewards.append(symexp(model.reward(imag_feat).squeeze(-1)))
        values.append(model.value(imag_feat).squeeze(-1))
        discounts.append(cont)
        logps.append(dist.log_prob(action))
        entropies.append(dist.entropy())

    imag_feats = torch.stack(imag_feats, 0)
    rewards = torch.stack(rewards, 0)
    values = torch.stack(values, 0)
    discounts = torch.stack(discounts, 0)
    bootstrap = model.value(model.feat(state)).squeeze(-1)
    targets = lambda_return(rewards, values, discounts, bootstrap, lambda_)
    return imag_feats, targets, values, torch.stack(logps, 0), torch.stack(entropies, 0)


class ReturnNormalizer:
    """DreamerV3 percentile-based return normalization (5th-95th)."""

    def __init__(self, decay=0.99):
        self.decay = decay
        self.low = 0.0
        self.high = 1.0

    def update(self, returns):
        # Track robust running bounds from imagined return distribution.
        # Using quantiles instead of mean/std avoids a few very good/bad rollouts
        # dominating actor gradients, which is common early in model learning.
        with torch.no_grad():
            flat = returns.detach().flatten()
            low = torch.quantile(flat, 0.05).item()
            high = torch.quantile(flat, 0.95).item()
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high

    def scale(self, advantage):
        # Advantage normalization is done by inter-quantile range (high-low)
        # rather than variance. This preserves sign/ranking while adapting
        # gradient scale to changing return magnitudes over training.
        denom = max(self.high - self.low, 1.0)
        return advantage / denom


def preprocess_frame(frame):
    frame = np.ascontiguousarray(np.asarray(frame))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame = F.interpolate(frame, size=(64, 64), mode="bilinear", align_corners=False)
    return frame.squeeze(0).byte().numpy()


def train(args):
    dev = device()
    env = gym.make(args.env, render_mode="rgb_array")
    action_dim = env.action_space.n
    model = DreamerV3(action_dim).to(dev)

    world_params = list(model.encoder.parameters()) + list(model.rssm.parameters()) + list(model.decoder.parameters())
    world_params += list(model.reward.parameters()) + list(model.cont.parameters())
    world_opt = optim.Adam(world_params, lr=args.world_lr, eps=1e-8)
    actor_opt = optim.Adam(model.actor.parameters(), lr=args.actor_lr, eps=1e-5)
    value_opt = optim.Adam(model.value.parameters(), lr=args.value_lr, eps=1e-5)
    replay = ReplayBuffer(capacity=args.replay_capacity)
    return_norm = ReturnNormalizer()
    world_losses, actor_losses, value_losses = [], [], []

    obs, _ = env.reset()
    obs = preprocess_frame(obs)
    # Phase 1: replay prefill.
    # Dreamer learns dynamics from replayed trajectories first; without this,
    # imagined rollouts would come from an untrained world model and produce
    # noisy policy/value targets.
    for step in range(args.prefill):
        action = env.action_space.sample()
        nxt, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        replay.add(obs, action, rew, float(done))
        obs = preprocess_frame(nxt if not done else env.reset()[0])
        if step % 2000 == 0 and step > 0:
            print(f"Prefill: {step}/{args.prefill}")

    for update in range(args.updates):
        obs_b, act_b, rew_b, done_b = replay.sample(args.batch_size, args.seq_len)
        obs_b = obs_b.to(dev)
        act_b = act_b.to(dev)
        rew_b = rew_b.to(dev)
        done_b = done_b.to(dev)

        state = model.rssm.init_state(args.batch_size, dev)
        posts, priors, feats = [], [], []
        # Phase 2a: world-model update from replay sequences.
        # This pass consumes real replay transitions (pixels/actions/rewards/dones)
        # and updates encoder + RSSM + decoder + reward/continuation heads.
        # done masking (inside the loop) prevents latent state from crossing
        # episode boundaries when sequence chunks include terminals.
        for t in range(args.seq_len):
            action_oh = F.one_hot(act_b[:, t], action_dim).float()
            embed = model.encoder(obs_b[:, t])
            state = mask_state(state, done_b[:, t - 1]) if t > 0 else state
            post, prior = model.rssm.observe_step(state, action_oh, embed)
            post = {**post, "probs": unimix_probs(post["probs"], mix=0.01)}
            prior = {**prior, "probs": unimix_probs(prior["probs"], mix=0.01)}
            feat = model.feat(post)
            posts.append(post)
            priors.append(prior)
            feats.append(feat)
            state = post

        feats = torch.stack(feats, 0)
        recon = model.decoder(feats.reshape(-1, feats.shape[-1]))
        target_img = obs_b[:, 1:].transpose(0, 1).reshape(-1, 3, 64, 64).float() / 255.0
        obs_loss = F.mse_loss(torch.sigmoid(recon), target_img)

        reward_pred = model.reward(feats).squeeze(-1)
        # Regress in symlog space to reduce sensitivity to reward spikes while
        # keeping small rewards near-linear around zero.
        reward_loss = F.mse_loss(reward_pred, symlog(rew_b.transpose(0, 1)))
        cont_logits = model.cont(feats).squeeze(-1)
        cont_target = 1.0 - done_b.transpose(0, 1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        # DreamerV3 KL split (dyn/rep):
        # - dyn KL: improves transition model by moving prior toward posterior
        #   statistics inferred from observations (representation target detached).
        # - rep KL: shapes encoder/posterior to stay aligned with model prior
        #   (prior target detached), preventing representation drift.
        # Separate weights let dynamics and representation be regularized at
        # different strengths, instead of one symmetric KL term.
        kl_terms = [DiscreteRSSM.kl_balanced(p, r, alpha=0.5) for p, r in zip(posts, priors)]
        dyn_kl = torch.stack([t[1] for t in kl_terms], 0)
        rep_kl = torch.stack([t[2] for t in kl_terms], 0)
        dyn_loss = free_nats_loss(dyn_kl, 1.0)
        rep_loss = free_nats_loss(rep_kl, 1.0)
        world_loss = obs_loss + reward_loss + cont_loss + 0.5 * dyn_loss + 0.1 * rep_loss

        world_opt.zero_grad()
        world_loss.backward()
        nn.utils.clip_grad_norm_(world_params, 100.0)
        world_opt.step()

        # Phase 2b: behavior learning on imagined latent rollouts.
        # Starting from replay-inferred posterior state, roll forward using RSSM
        # imagination and train actor/value without new environment interaction.
        imag_feats, imag_target, imag_value, logp, entropy = imagine_behavior(
            model, posts[-1], horizon=args.horizon, action_dim=action_dim, gamma=0.997, lambda_=0.95
        )
        # Normalize imagined advantages with robust running return scale before
        # policy-gradient weighting, which stabilizes actor updates over training.
        return_norm.update(imag_target)
        advantage = return_norm.scale(imag_target - imag_value)
        actor_loss = -(logp * advantage.detach() + 3e-4 * entropy).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 100.0)
        actor_opt.step()

        value_pred = model.value(imag_feats.detach()).squeeze(-1)
        value_loss = F.mse_loss(value_pred, imag_target.detach())
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
    plt.title("Dreamer V3 Training Losses")
    plt.legend()
    plt.grid()
    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out)
    print(f"Training curve saved: {out}")


def evaluate_and_record(model, env_name, save_dir, dev, max_steps=1000):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    obs = preprocess_frame(obs)
    state = model.rssm.init_state(batch=1, device=dev)
    prev_action = torch.zeros(1, env.action_space.n, device=dev)
    frames = []
    total_reward = 0.0
    done = False
    steps = 0

    # FIRE to start Breakout.
    obs, rew, term, trunc, _ = env.step(1)
    total_reward += rew
    done = term or trunc
    obs = preprocess_frame(obs if not done else env.reset()[0])
    prev_action = F.one_hot(torch.tensor([1], device=dev), env.action_space.n).float()

    with torch.no_grad():
        while not done and steps < max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=dev).unsqueeze(0)
            embed = model.encoder(obs_t)
            post, _ = model.rssm.observe_step(state, prev_action, embed)
            feat = model.feat(post)
            logits = model.actor(feat)
            action = torch.argmax(logits, dim=-1)
            nxt, rew, term, trunc, _ = env.step(int(action.item()))
            done = term or trunc
            total_reward += rew
            obs = preprocess_frame(nxt if not done else env.reset()[0])
            state = post
            prev_action = F.one_hot(action, env.action_space.n).float()
            steps += 1

    env.close()
    gif_path = os.path.join(save_dir, "dreamer_v3_agent.gif")
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Evaluation GIF saved: {gif_path}")
    print(f"Evaluation reward: {total_reward:.2f}")


def build_args():
    p = argparse.ArgumentParser(description="Dreamer V3 (paper architecture) in PyTorch")
    p.add_argument("--env", type=str, default="ALE/Breakout-v5")
    p.add_argument("--prefill", type=int, default=200_000)
    p.add_argument("--updates", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--replay-capacity", type=int, default=2_000_000)
    p.add_argument("--world-lr", type=float, default=1e-4)
    p.add_argument("--actor-lr", type=float, default=3e-5)
    p.add_argument("--value-lr", type=float, default=3e-5)
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    dev = device()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "dreamer_v3", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all outputs to: {save_dir}")

    model, metrics = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    plot_metrics(metrics, save_dir)
    evaluate_and_record(model, args.env, save_dir, dev)
