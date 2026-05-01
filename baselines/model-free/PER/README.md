# PER DDQN (Breakout)

Minimal guide to run the Prioritized Experience Replay + Double DQN script.

## Run

From the repository root:

```bash
python3 baselines/model-free/PER/per_ddqn.py
```

## Inputs / Arguments

- Script currently has no CLI flags.
- Core defaults are defined in code:
  - env: `ALE/Breakout-v5`
  - `max_frames=1000000`
  - `batch_size=32`
  - optimizer: DeepMind-style RMSProp (`lr=2.5e-4`, `alpha=0.95`, `momentum=0.95`, `eps=0.01`)
  - replay warmup: `50000` transitions
  - PER: `alpha=0.6`, `beta_start=0.4` annealed to `1.0` over `250000` agent decisions
  - epsilon schedule: `1.0 -> 0.1` over `250000` agent decisions (`1M` Atari frames with `frame_skip=4`)
  - target update frequency: every `10000` frames

## Expected console output

You should see logs similar to:

```text
Using device: cuda
Saving all results to: /home/administrator/baselines/results/per_ddqn/run_YYYYMMDD_HHMMSS
Starting Prioritized Double DQN (PER DDQN) Training for 1000000 frames...
Frame: 2000/1000000 | Epsilon: 1.00 | Avg Reward (Last 20): ...
...
Model saved to /home/administrator/baselines/results/per_ddqn/run_YYYYMMDD_HHMMSS/model.pth
Training curve saved as '/home/administrator/baselines/results/per_ddqn/run_YYYYMMDD_HHMMSS/training_curve.png'
Evaluating agent and saving video to /home/administrator/baselines/results/per_ddqn/run_YYYYMMDD_HHMMSS/breakout_per_ddqn_agent.gif
Evaluation finished. Total Reward: ...
Saved successfully!
```

Device line may be `cuda`, `mps`, or `cpu` depending on hardware.

## Output artifacts

Each run creates a timestamped folder under:

`results/per_ddqn/run_<timestamp>/`

with:

- `model.pth` (trained network weights),
- `training_curve.png` (episode rewards plot),
- `breakout_per_ddqn_agent.gif` (evaluation rollout).
