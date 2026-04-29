# `learned_dynamics_mpc.py` Implementation Walkthrough

This document explains the exact pipeline implemented in `learned_dynamics_mpc.py`.

## 1) Dynamics ensemble

- Uses **5 separate `DynamicsModel` networks**.
- Each model predicts **normalized delta state** (`next_state - state`) from `(state, action)`.
- Architecture per model:
  - Input: `state_dim + action_dim`
  - MLP: `128 -> 128` with ReLU
  - Output: `state_dim` (delta prediction)
- Each model has its own Adam optimizer (`lr=1e-3`).

Ensemble prediction is the **mean** of model outputs after denormalization.

## 2) Transition normalization

`TransitionNormalizer` computes replay-wide statistics:

- State mean/std
- Action mean/std
- Delta mean/std

It standardizes inputs/targets for model training and denormalizes predicted deltas during rollouts. This stabilizes optimization and keeps scale consistent between training and planning.

## 3) Data pipeline and training schedule

- Replay buffer stores `(state, action, next_state)` with capacity `100000`.
- Seed data: `2000` random transitions from `Pendulum-v1`.
- Initial model pretraining: `epochs=100`, `batch_size=256`.
- Online phase: `25` episodes of MPC control.
- After each episode:
  - add collected transitions to replay
  - retrain dynamics ensemble for `10` epochs (`batch_size=256`)
  - log one-step and multi-step errors.

## 4) CEM planner details

`CEMPlanner` parameters in this script:

- `num_sequences (K) = 512` during training (`1024` at evaluation)
- `horizon (H) = 20`
- `elite_frac = 0.1`
- `iterations = 4`
- `gamma = 0.99`

Per planning step:

1. Sample `H x K x action_dim` action candidates from Gaussian.
2. Clip to environment action bounds.
3. Roll out each sequence with learned dynamics.
4. Score by discounted sum of `pendulum_reward_fn`.
5. Keep top elites; refit Gaussian mean/std.
6. Return first action of best sequence.

The planner re-runs this optimization at every environment step (receding horizon MPC).

## 5) Rollout scoring

`rollout_return()` does vectorized imaginary rollouts:

- Starts from current real state, copied across all `K` candidates.
- Applies predicted delta: `next = state + predicted_delta`.
- Renormalizes Pendulum observation `(cos(theta), sin(theta))` to unit norm.
- Computes reward via `pendulum_reward_fn`:
  - `-(theta^2 + 0.1 * theta_dot^2 + 0.001 * action^2)`
- Applies geometric discount with `gamma`.

## 6) Multistep validation

`validate_multistep(..., horizon=5)` checks compounding model error:

- Randomly chooses replay starting points.
- Unrolls predicted state forward for 5 steps using recorded actions.
- Compares predicted final state to replay target final state.
- Reports mean MSE over sampled starts.

This complements one-step training MSE and better reflects planning-time model quality.

## 7) Outputs produced by this pipeline

- Saved ensemble weights (`dynamics_ensemble.pth`)
- Reward curve plot (`training_curve.png`)
- Evaluation GIF (`pendulum_mpc_agent.gif`)

All outputs are stored in `results/mpc/run_<timestamp>/`.
