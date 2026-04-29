# Dreamer V1

Continuous-action Dreamer baseline with a continuous RSSM world model.

## Run

From repo root:

```bash
python baselines/model-based/Dreamer-v1/dreamer_v1.py
```

Research-style custom run:

```bash
python baselines/model-based/Dreamer-v1/dreamer_v1.py \
  --env dm_control/walker-walk-v0 \
  --prefill 50000 \
  --updates 20000 \
  --batch-size 64 \
  --seq-len 64 \
  --horizon 15 \
  --replay-capacity 500000 \
  --world-lr 6e-4 \
  --actor-lr 8e-5 \
  --value-lr 8e-5
```

## Inputs / Arguments

- `--env` (default: `dm_control/walker-walk-v0`)
- `--prefill` (default: `50000`)
- `--updates` (default: `20000`)
- `--batch-size` (default: `64`)
- `--seq-len` (default: `64`)
- `--horizon` (default: `15`)
- `--replay-capacity` (default: `500000`)
- `--world-lr` (default: `6e-4`)
- `--actor-lr` (default: `8e-5`)
- `--value-lr` (default: `8e-5`)

## Environment Notes

- Requires: `torch`, `gymnasium`, `shimmy`, `numpy`, `matplotlib`, `imageio`.
- Uses `render_mode="rgb_array"` and resizes observations to `64x64`.
- On headless Linux, script sets EGL-related env vars for rendering.

## Outputs

Each run writes:

`results/dreamer_v1/run_<timestamp>/`

- `model.pth`
- `training_curve.png`
- `dreamer_v1_agent.gif`
