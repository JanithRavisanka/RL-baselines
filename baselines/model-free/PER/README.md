# PER DDQN (Breakout)

Minimal guide to run the Prioritized Experience Replay + Double DQN script.

## Run

From the repository root:

```bash
python3 baselines/model-free/PER/per_ddqn.py
```

## Expected console output

You should see logs similar to:

```text
Using device: cuda
Saving all results to: /home/administrator/baselines/results/per_ddqn/run_YYYYMMDD_HHMMSS
Starting Prioritized Double DQN (PER DDQN) Training for 500000 frames...
Frame: 2000/500000 | Epsilon: 0.99 | Avg Reward (Last 20): ...
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
