# How `dyna_q.py` Implements Dyna-Q

This file applies tabular Dyna-Q to `CliffWalking-v1` (Gymnasium), with explicit separation between exploratory training returns and greedy evaluation.

## Main Structure

- **`DynaQAgent`**
  - `q_table`: NumPy array of shape `(state_dim, action_dim)`.
  - `model`: dictionary mapping `(state, action)` to `(reward, next_state, done)`.
  - `observed_states` + `observed_actions[state]`: supports sampling only previously seen state-action pairs during planning.
- **`train()`**: runs environment interaction and learning for `50000` episodes.
- **`evaluate_policy()`**: computes mean greedy return (`epsilon=0`) over short evaluation episodes.
- **`plot_rewards()`**: writes training/evaluation curve image.
- **`evaluate_and_record()`**: renders greedy rollout to GIF.

## Real + Planning Updates

`DynaQAgent.learn(state, action, reward, next_state, done)` performs all Dyna-Q substeps:

1. **Direct Q-learning update** from real transition.
2. **Model update**: saves `(reward, next_state, done)` in `self.model[(state, action)]`.
3. **Planning loop** (`for _ in range(self.planning_steps)`):
   - sample a previously observed state then an action observed in that state,
   - fetch simulated transition from the model,
   - apply the same Q-learning backup.

This uses uniform random sampling over historically observed pairs (via the state/action bookkeeping lists).

## Terminal Handling

Terminal status is treated as `done = terminated or truncated` (Gymnasium API).

- In both real and simulated updates, target is:
  - `reward` if `done` is true,
  - otherwise `reward + gamma * max_a Q(next_state, a)`.
- Evaluation routines cap episode length (100 steps) to avoid non-terminating rollouts.

## Training and Evaluation Curves

`train()` tracks:

- **`episode_rewards`**: per-episode exploratory return under decaying epsilon.
- **`eval_rewards`**: every 50 episodes, mean return from `evaluate_policy()` with epsilon temporarily forced to 0.

`plot_rewards()` then saves:

- raw exploratory return curve,
- 50-episode moving average (red),
- periodic greedy eval points (green markers).

This makes policy quality easier to read than exploratory reward alone.
