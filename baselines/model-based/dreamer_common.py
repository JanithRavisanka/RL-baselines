import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def lambda_return(reward, value, discount, bootstrap, lam):
    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    target = reward + discount * ((1 - lam) * next_values)
    returns = []
    acc = bootstrap
    for t in reversed(range(len(target))):
        acc = target[t] + discount[t] * lam * acc
        returns.append(acc)
    returns.reverse()
    return torch.stack(returns, dim=0)


def free_nats_loss(kl: torch.Tensor, free_nats: float) -> torch.Tensor:
    """Only penalize KL above the free-nats threshold."""
    return torch.clamp(kl - free_nats, min=0.0).mean()


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2),
            nn.ELU(),
        )

    def forward(self, x):
        x = x.float() / 255.0
        h = self.net(x)
        return h.flatten(start_dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, depth=48, emb=1536):
        super().__init__()
        self.fc = nn.Linear(emb, depth * 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(depth * 32, depth * 4, 5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(depth, out_channels, 6, stride=2),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], -1, 1, 1)
        return self.deconv(h)


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=400, layers=3, act=nn.ELU):
        super().__init__()
        net = []
        dim = in_dim
        for _ in range(layers):
            net.extend([nn.Linear(dim, hidden), act()])
            dim = hidden
        net.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ContinuousRSSM(nn.Module):
    def __init__(self, action_dim, deter=200, stoch=30, hidden=200, embed_dim=1536):
        super().__init__()
        self.action_dim = action_dim
        self.deter = deter
        self.stoch = stoch
        self.gru = nn.GRUCell(stoch + action_dim, deter)
        self.prior = MLPHead(deter, 2 * stoch, hidden=hidden, layers=2)
        self.post = MLPHead(deter + embed_dim, 2 * stoch, hidden=hidden, layers=2)

    def init_state(self, batch, device):
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": torch.zeros(batch, self.stoch, device=device),
            "mean": torch.zeros(batch, self.stoch, device=device),
            "std": torch.ones(batch, self.stoch, device=device),
        }

    def _dist(self, stats):
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = F.softplus(log_std) + 1e-4
        return mean, std

    def observe_step(self, prev, action, embed):
        x = torch.cat([prev["stoch"], action], dim=-1)
        deter = self.gru(x, prev["deter"])
        p_stats = self.prior(deter)
        p_mean, p_std = self._dist(p_stats)
        h = torch.cat([deter, embed], dim=-1)
        q_stats = self.post(h)
        q_mean, q_std = self._dist(q_stats)
        stoch = q_mean + q_std * torch.randn_like(q_std)
        post = {"deter": deter, "stoch": stoch, "mean": q_mean, "std": q_std}
        prior = {"deter": deter, "stoch": p_mean, "mean": p_mean, "std": p_std}
        return post, prior

    def imagine_step(self, prev, action):
        x = torch.cat([prev["stoch"], action], dim=-1)
        deter = self.gru(x, prev["deter"])
        p_mean, p_std = self._dist(self.prior(deter))
        stoch = p_mean + p_std * torch.randn_like(p_std)
        return {"deter": deter, "stoch": stoch, "mean": p_mean, "std": p_std}

    @staticmethod
    def kl(post, prior):
        p = torch.distributions.Normal(post["mean"], post["std"])
        q = torch.distributions.Normal(prior["mean"], prior["std"])
        return torch.distributions.kl_divergence(p, q).sum(-1)


class DiscreteRSSM(nn.Module):
    def __init__(self, action_dim, deter=600, classes=32, stoch=32, hidden=600, embed_dim=1536):
        super().__init__()
        self.action_dim = action_dim
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.gru = nn.GRUCell(stoch * classes + action_dim, deter)
        self.prior = MLPHead(deter, stoch * classes, hidden=hidden, layers=2)
        self.post = MLPHead(deter + embed_dim, stoch * classes, hidden=hidden, layers=2)

    def init_state(self, batch, device):
        z = torch.zeros(batch, self.stoch, self.classes, device=device)
        z[..., 0] = 1.0
        return {"deter": torch.zeros(batch, self.deter, device=device), "stoch": z}

    def _sample(self, logits):
        logits = logits.view(logits.shape[0], self.stoch, self.classes)
        probs = F.softmax(logits, dim=-1)
        sample = F.one_hot(torch.multinomial(probs.view(-1, self.classes), 1).squeeze(-1), self.classes).float()
        sample = sample.view(logits.shape[0], self.stoch, self.classes)
        sample = sample + probs - probs.detach()
        return sample, probs

    def observe_step(self, prev, action, embed):
        prev_flat = prev["stoch"].flatten(start_dim=1)
        x = torch.cat([prev_flat, action], dim=-1)
        deter = self.gru(x, prev["deter"])
        p_logits = self.prior(deter)
        q_logits = self.post(torch.cat([deter, embed], dim=-1))
        stoch, q_probs = self._sample(q_logits)
        p_stoch, p_probs = self._sample(p_logits)
        post = {"deter": deter, "stoch": stoch, "probs": q_probs}
        prior = {"deter": deter, "stoch": p_stoch, "probs": p_probs}
        return post, prior

    def imagine_step(self, prev, action):
        prev_flat = prev["stoch"].flatten(start_dim=1)
        x = torch.cat([prev_flat, action], dim=-1)
        deter = self.gru(x, prev["deter"])
        logits = self.prior(deter)
        stoch, probs = self._sample(logits)
        return {"deter": deter, "stoch": stoch, "probs": probs}

    @staticmethod
    def kl(post, prior):
        q = torch.distributions.Categorical(probs=post["probs"])
        p = torch.distributions.Categorical(probs=prior["probs"])
        return torch.distributions.kl_divergence(q, p).sum(-1)


def actor_loss(log_prob, advantage, entropy, entropy_scale=1e-3):
    return -(log_prob * advantage.detach() + entropy_scale * entropy).mean()


def value_loss(pred, target):
    return F.mse_loss(pred, target.detach())


@dataclass
class DreamerConfig:
    horizon: int
    gamma: float
    lambda_: float
    world_lr: float
    actor_lr: float
    value_lr: float
    kl_scale: float
    free_nats: float
    batch_size: int
    seq_len: int
    latent_dim: int
    deter_dim: int
    atari: bool = False
