# Dyna-Q Baseline (`CliffWalking-v1`)

This script trains a tabular Dyna-Q agent on Gymnasium's `CliffWalking-v1`, saves artifacts, and records a greedy policy rollout.

## Run

From repository root:

```bash
python baselines/model-based/Dyna-Q/dyna_q.py
```

No CLI arguments are defined in the current script. Hyperparameters are set in code defaults/usages:

- agent: `alpha=0.1`, `gamma=0.99`, `epsilon=0.1`, `planning_steps=50`
- training episodes: `50000`
- epsilon decay per episode: `epsilon = max(0.01, epsilon * 0.99995)`

## Dependencies

The script imports:

- `gymnasium`
- `numpy`
- `matplotlib`
- `imageio`

## What Gets Generated

On each run, outputs are written to:

- `results/dyna_q/run_<YYYYMMDD_HHMMSS>/`

Files created in that run folder:

- `q_table.npy` - learned tabular action-values.
- `training_curve.png` - exploratory returns, moving average, and periodic greedy eval.
- `cliffwalking_agent.gif` - rendered greedy evaluation rollout.

The script also prints progress every 50 episodes with:

- exploratory train average over last 50 episodes,
- greedy evaluation average from `evaluate_policy()`.

## Notes for Newcomers

- Training returns can stay noisy because exploration remains active.
- Greedy evaluation (`epsilon=0`) is logged separately to show actual policy quality.
- Planning updates happen every real step (`planning_steps=50`), which is the sample-efficiency core of Dyna-Q.
