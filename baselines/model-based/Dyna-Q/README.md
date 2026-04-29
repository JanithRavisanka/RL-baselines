# Dyna-Q Baseline (`CliffWalking-v1`)

This script trains a tabular Dyna-Q agent on Gymnasium's `CliffWalking-v1`, saves artifacts, and records a greedy policy rollout.

## Run

From repository root:

```bash
python baselines/model-based/Dyna-Q/dyna_q.py
```

Research-style custom run:

```bash
python baselines/model-based/Dyna-Q/dyna_q.py \
  --num-episodes 100000 \
  --planning-steps 100 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon-start 0.2 \
  --epsilon-end 0.01 \
  --epsilon-decay 0.99997 \
  --eval-interval 100
```

## Inputs / Arguments

- `--num-episodes` (default: `100000`)
- `--planning-steps` (default: `100`)
- `--alpha` (default: `0.1`)
- `--gamma` (default: `0.99`)
- `--epsilon-start` (default: `0.2`)
- `--epsilon-end` (default: `0.01`)
- `--epsilon-decay` (default: `0.99997`)
- `--eval-interval` (default: `100`)

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

The script also prints progress every `eval-interval` episodes with:

- exploratory train average over last `eval-interval` episodes,
- greedy evaluation average from `evaluate_policy()`.

## Notes for Newcomers

- Training returns can stay noisy because exploration remains active.
- Greedy evaluation (`epsilon=0`) is logged separately to show actual policy quality.
- Planning updates happen every real step (`planning_steps`), which is the sample-efficiency core of Dyna-Q.
