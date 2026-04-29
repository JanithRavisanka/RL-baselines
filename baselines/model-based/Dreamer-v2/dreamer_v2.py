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
from dreamer_common import ConvDecoder, ConvEncoder, DiscreteRSSM, DreamerConfig, MLPHead, free_nats_loss, lambda_return  # noqa: F401


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
        # We train the world model on short contiguous trajectories instead of
        # single transitions, because RSSM needs temporal context to learn
        # latent dynamics and to align posterior/prior over multiple steps.
        #
        # Returned shapes:
        # - obs:  [B, seq+1, C, H, W] (extra frame gives reconstruction target t+1)
        # - act:  [B, seq]
        # - rew:  [B, seq]
        # - done: [B, seq]
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


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


def mask_state(state, done):
    # "done" marks terminal transition at previous step. We zero latent state
    # before continuing unroll so the next step starts from a clean state for
    # episodes that ended inside the sampled sequence.
    #
    # This avoids leaking information across episode boundaries when replay
    # sequences are sampled blindly from a global buffer.
    alive = 1.0 - done
    return {
        "deter": state["deter"] * alive.unsqueeze(-1),
        "stoch": state["stoch"] * alive.view(-1, 1, 1),
        "probs": state.get("probs", state["stoch"]) * alive.view(-1, 1, 1),
    }


def imagine_behavior(model, start, horizon, action_dim, gamma, lambda_):
    """
    Discrete-action imagined rollout with straight-through one-hot actions.

    Distinctive point:
    - Actor samples categorical actions.
    - We feed one-hot actions into RSSM prior with ST estimator so actor gradients
      can influence imagined trajectory optimization.
    - This keeps forward dynamics discrete (sampled action is actually executed
      in imagination), while backprop sees a differentiable path through the
      policy probabilities.
    """
    state = detach_state(start)
    imag_feats, rewards, values, discounts, logps, entropies = [], [], [], [], [], []
    for _ in range(horizon):
        feat = model.feat(state)
        dist = model.actor_dist(feat)
        action = dist.sample()
        probs = dist.probs
        action_oh = F.one_hot(action, action_dim).float()
        # Straight-through trick:
        # - Forward pass uses sampled one-hot action (categorical control).
        # - Backward pass uses gradient of probs as if action were continuous.
        # This allows policy improvement through imagined latent rollouts.
        action_oh = action_oh + probs - probs.detach()
        state = model.rssm.imagine_step(state, action_oh)
        imag_feat = model.feat(state)
        imag_feats.append(imag_feat)
        rewards.append(model.reward(imag_feat).squeeze(-1))
        values.append(model.value(imag_feat).squeeze(-1))
        discounts.append(torch.full_like(rewards[-1], gamma))
        logps.append(dist.log_prob(action))
        entropies.append(dist.entropy())

    imag_feats = torch.stack(imag_feats, 0)
    rewards = torch.stack(rewards, 0)
    values = torch.stack(values, 0)
    discounts = torch.stack(discounts, 0)
    bootstrap = model.value(model.feat(state)).squeeze(-1)
    # Lambda returns on imagined trajectories provide low-variance, limited-bias
    # targets for both actor advantage and critic regression.
    targets = lambda_return(rewards, values, discounts, bootstrap, lambda_)
    return imag_feats, targets, values, torch.stack(logps, 0), torch.stack(entropies, 0)


def preprocess_frame(frame):
    frame = np.ascontiguousarray(np.asarray(frame))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame = F.interpolate(frame, size=(64, 64), mode="bilinear", align_corners=False)
    return frame.squeeze(0).byte().numpy()


def train(args):
    cfg = DreamerConfig(
        horizon=args.horizon,
        gamma=0.997,
        lambda_=0.95,
        world_lr=args.world_lr,
        actor_lr=args.actor_lr,
        value_lr=args.value_lr,
        kl_scale=1.0,
        free_nats=1.0,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
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
    replay = ReplayBuffer(capacity=args.replay_capacity)
    world_losses, actor_losses, value_losses = [], [], []

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
        # World-model learning from replay sequences:
        # 1) Encode each frame into embeddings.
        # 2) Recurrently infer posterior z_t from embed_t and prior from
        #    previous latent/action.
        # 3) Decode latent features back to pixels and rewards.
        #
        # Sequence training (instead of single-step) is essential here: RSSM
        # learns temporal consistency and the recurrent deterministic state.
        for t in range(cfg.seq_len):
            action_oh = F.one_hot(act_b[:, t], action_dim).float()
            embed = model.encoder(obs_b[:, t])
            # If previous transition ended episode, zero latent before stepping.
            # done_b[:, t-1] is used because it terminates transition into t.
            state = mask_state(state, done_b[:, t - 1]) if t > 0 else state
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
        # DreamerV2 KL balancing (alpha=0.8):
        # stronger pressure on prior to match posterior than the reverse.
        # Intuition:
        # - Posterior has image evidence (more accurate for current step).
        # - Prior must become predictive for imagination.
        # Balanced KL uses asymmetric gradient flow so prior learns to chase
        # posterior without over-constraining posterior quality.
        kl_terms = [DiscreteRSSM.kl_balanced(p, r, alpha=0.8) for p, r in zip(posts, priors)]
        kl_total = torch.stack([t[0] for t in kl_terms], 0)
        # Free-nats keeps small KL deviations unpenalized, preventing early
        # over-regularization and posterior collapse in latent dynamics.
        kl_loss = free_nats_loss(kl_total, cfg.free_nats)
        world_loss = recon_loss + reward_loss + cfg.kl_scale * kl_loss

        world_opt.zero_grad()
        world_loss.backward()
        nn.utils.clip_grad_norm_(world_params, 100.0)
        world_opt.step()

        # Behavior learning over imagined latent futures:
        # - Roll out RSSM prior from last posterior state (no new pixels).
        # - Actor optimized with REINFORCE-style objective using imagined
        #   advantages, plus entropy bonus for exploration.
        # - Value regressed to lambda-return targets from same imagination.
        imag_feats, imag_target, imag_value, log_prob, entropy = imagine_behavior(
            model, posts[-1], cfg.horizon, action_dim, cfg.gamma, cfg.lambda_
        )
        advantage = imag_target - imag_value
        actor_loss = -(log_prob * advantage.detach() + 3e-4 * entropy).mean()

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
    plt.title("Dreamer V2 Training Losses")
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

    # ALE Breakout requires FIRE to launch the ball after reset; evaluation
    # follows that convention so the policy is judged on active gameplay.
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
            # Evaluation is deterministic (argmax) rather than sampling, so
            # reported reward reflects greedy policy performance.
            action = torch.argmax(logits, dim=-1)
            nxt, rew, term, trunc, _ = env.step(int(action.item()))
            done = term or trunc
            total_reward += rew
            obs = preprocess_frame(nxt if not done else env.reset()[0])
            state = post
            # Keep previous action as one-hot because RSSM transition model
            # expects the same action representation as in training.
            prev_action = F.one_hot(action, env.action_space.n).float()
            steps += 1

    env.close()
    gif_path = os.path.join(save_dir, "dreamer_v2_agent.gif")
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Evaluation GIF saved: {gif_path}")
    print(f"Evaluation reward: {total_reward:.2f}")


def build_args():
    p = argparse.ArgumentParser(description="Dreamer V2 (paper architecture) in PyTorch")
    p.add_argument("--env", type=str, default="ALE/Breakout-v5")
    p.add_argument("--prefill", type=int, default=100_000)
    p.add_argument("--updates", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--replay-capacity", type=int, default=1_000_000)
    p.add_argument("--world-lr", type=float, default=3e-4)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--value-lr", type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    dev = device()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "dreamer_v2", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all outputs to: {save_dir}")

    model, metrics = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    plot_metrics(metrics, save_dir)
    evaluate_and_record(model, args.env, save_dir, dev)
