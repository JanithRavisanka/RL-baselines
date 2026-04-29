# Implementation Notes for This Folder

This folder has two scripts:
- `actor_critic.py` (discrete control on `CartPole-v1`)
- `actor_critic_continuous.py` (continuous control on `Pendulum-v1`)

Both implement episodic actor-critic with Monte Carlo returns and a shared actor/critic network trunk.

## How `actor_critic.py` maps to canonical actor-critic

- **Policy head (actor):** softmax over discrete actions, sampled with `Categorical`.
- **Value head (critic):** scalar `V(s)`.
- **Data collection:** runs full episodes, stores `log_prob(a_t|s_t)`, `V(s_t)`, rewards.
- **Targets:** discounted Monte Carlo returns `G_t`, then normalized per episode.
- **Advantage:** `A_t = G_t - V(s_t)` (implemented using `value.item()` for baseline term).
- **Losses:**
  - actor: `-log_prob * A_t`
  - critic: Smooth L1 between `V(s_t)` and `G_t`
- **Update:** single optimizer step on sum of actor + critic losses (shared trunk updated by both).

## How `actor_critic_continuous.py` maps to canonical actor-critic

- **Policy head (actor):** outputs Gaussian parameters (`mu`, `sigma`) and samples with `Normal`.
- **Action handling:** sampled actions are clipped to environment bounds before `env.step`.
- **Value head (critic):** scalar `V(s)`.
- **Targets/advantage/loss shape:** same pattern as discrete script:
  - normalized Monte Carlo returns
  - `A_t = G_t - V(s_t)`
  - actor loss `-log_prob * A_t`
  - critic Smooth L1 regression
- **Extra stability step:** gradient clipping (`max_norm=1.0`).

## Simplifications vs more advanced actor-critic methods

- Uses **full-episode Monte Carlo** returns, not n-step TD or GAE.
- No entropy bonus in loss (so exploration is only from stochastic sampling).
- No PPO/TRPO-style clipped trust region objective.
- No replay buffer, target networks, or parallel rollout workers.
- Continuous policy uses direct Gaussian sampling + clipping (no tanh-logprob correction pipeline).
- Hyperparameters are minimal and mostly hard-coded inside scripts.

So this is a clear educational actor-critic baseline rather than a production-grade on-policy algorithm stack.
