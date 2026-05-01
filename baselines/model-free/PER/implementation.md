# `per_ddqn.py` Implementation Notes

This file implements Atari Breakout training with **PER + DDQN**. The network and environment setup follow a standard DQN-style pipeline, while replay logic is replaced with prioritized sampling.

## 1) `SumTree`

`SumTree` is used to store replay priorities efficiently:

- `tree`: array of length `2 * capacity - 1` (binary tree of cumulative sums),
- `data`: ring buffer of transitions,
- `add(p, data)`: writes transition at current pointer and updates leaf priority,
- `update(idx, p)`: updates one leaf and propagates change upward,
- `get(s)`: retrieves a leaf by cumulative-priority lookup.

This gives `O(log N)` sampling and updates.

## 2) Priority assignment and sampling

`PrioritizedReplayBuffer` behavior:

- New transitions are inserted with current maximum priority (or `1.0` initially) so they are sampled early.
- Priority exponent uses `alpha=0.6`.
- Sampling is stratified: total priority is split into `batch_size` segments, and one value is sampled per segment for lower variance.

Sampling returns transitions, tree indices, and IS weights.

## 3) Beta annealing (IS correction strength)

In training:

- `beta` starts at `0.4`,
- increases linearly to `1.0` over `250000` agent decisions.

Higher beta later in training applies stronger bias correction when policy/value estimates become more sensitive.

## 4) Weighted loss

For each sampled batch:

- DDQN target uses online argmax and target-network evaluation.
- Elementwise Huber loss is computed with `reduction='none'`.
- Final loss is `(weights * elementwise_loss).mean()`.

This preserves PER's prioritization while correcting for non-uniform sampling.

## 5) Priority updates

After TD target computation:

- absolute TD errors are computed per sample,
- priorities are updated with:
  - `priority = (|td_error| + epsilon) ** alpha`,
  - `epsilon=0.01` prevents zero priority.

Updating priorities every optimization step keeps replay aligned with current learning signal.

## 6) Optimizer

The Q-network uses the DeepMind DQN RMSProp variant:

- learning rate `2.5e-4`,
- squared-gradient decay `0.95`,
- momentum `0.95`,
- epsilon `0.01` inside the square root.

This is intentionally not `torch.optim.RMSprop(..., eps=0.01)`, because PyTorch applies epsilon outside the square root and does not match the DeepMind DQN momentum update by default.
