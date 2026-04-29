# Dreamer V2

Discrete-latent Dreamer baseline for Atari-style tasks.

## Run

From repo root:

```bash
python baselines/model-based/Dreamer-v2/dreamer_v2.py
```

Research-style custom run:

```bash
python baselines/model-based/Dreamer-v2/dreamer_v2.py \
  --env ALE/Breakout-v5 \
  --prefill 100000 \
  --updates 30000 \
  --batch-size 32 \
  --seq-len 64 \
  --horizon 15 \
  --replay-capacity 1000000 \
  --world-lr 3e-4 \
  --actor-lr 1e-4 \
  --value-lr 1e-4
```

## Inputs / Arguments

- `--env` (default: `ALE/Breakout-v5`)
- `--prefill` (default: `100000`)
- `--updates` (default: `30000`)
- `--batch-size` (default: `32`)
- `--seq-len` (default: `64`)
- `--horizon` (default: `15`)
- `--replay-capacity` (default: `1000000`)
- `--world-lr` (default: `3e-4`)
- `--actor-lr` (default: `1e-4`)
- `--value-lr` (default: `1e-4`)

## Environment Notes

- Requires: `torch`, `gymnasium`, `ale-py`, `numpy`, `matplotlib`, `imageio`.
- Uses `rgb_array` frames; preprocessing resizes to `64x64`.
- Evaluation uses an initial Breakout `FIRE` action.

## Outputs

Each run writes:

`results/dreamer_v2/run_<timestamp>/`

- `model.pth`
- `training_curve.png`
- `dreamer_v2_agent.gif`

