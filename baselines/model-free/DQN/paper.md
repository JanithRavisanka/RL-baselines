# Deep Q-Network (DQN) - Canonical Method Notes

This file summarizes the core ideas behind the original DQN algorithm (Mnih et al., 2015) in the form most Atari implementations use.

## 1) Experience Replay

- Store transitions `(s, a, r, s', done)` in a replay buffer.
- Train on random mini-batches sampled from that buffer, rather than only the newest transition.
- Why it matters:
  - breaks strong temporal correlation in online trajectories;
  - improves data efficiency by reusing older experience;
  - stabilizes optimization for nonlinear function approximators.

## 2) Target Network

- Maintain two Q-networks:
  - **online network** `Q_theta` (updated every gradient step),
  - **target network** `Q_theta-` (updated periodically by copying online weights).
- Bellman bootstrap targets use the lagged target network, which reduces moving-target instability.
- Typical update pattern: hard copy every fixed number of environment steps.

## 3) Bellman Target (1-step TD)

For each sampled transition:

- If not terminal:
  - `y = r + gamma * max_a' Q_theta-(s', a')`
- If terminal:
  - `y = r`

Then minimize the TD error between online prediction and target:

- prediction: `Q_theta(s, a)`
- target: `y`
- loss: commonly Huber (smooth L1) or MSE.

## 4) Epsilon-Greedy Exploration

- Behavior policy mixes exploration and exploitation:
  - with probability `epsilon`, choose a random action;
  - otherwise choose `argmax_a Q_theta(s, a)`.
- `epsilon` is annealed from high to lower values through training (e.g., 1.0 to 0.1).
- This keeps early exploration broad and later behavior more value-driven.

## 5) Practical Atari Conventions

Canonical DQN setups for Atari often also include:

- frame preprocessing (grayscale, resize to 84x84),
- frame stacking (typically 4 frames),
- reward clipping to `[-1, 1]`.

These are not the Bellman update itself, but they are part of the practical recipe that made DQN effective.
