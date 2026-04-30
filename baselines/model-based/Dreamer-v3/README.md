# Dreamer V3

Dreamer V3-style discrete world-model agent for Atari-like visual environments.

## Run

From repo root:

```bash
python baselines/model-based/Dreamer-v3/dreamer_v3.py
```

Research-style custom run:

```bash
python baselines/model-based/Dreamer-v3/dreamer_v3.py \
  --env ALE/Breakout-v5 \
  --prefill 200000 \
  --updates 50000 \
  --batch-size 32 \
  --seq-len 64 \
  --horizon 15 \
  --replay-capacity 2000000 \
  --world-lr 1e-4 \
  --actor-lr 3e-5 \
  --value-lr 3e-5 \
  --collect-interval 100 \
  --collect-steps 1000
```

## Inputs / Arguments

- `--env` (default: `ALE/Breakout-v5`)
- `--prefill` (default: `200000`)
- `--updates` (default: `50000`)
- `--batch-size` (default: `32`)
- `--seq-len` (default: `64`)
- `--horizon` (default: `15`)
- `--replay-capacity` (default: `2000000`)
- `--world-lr` (default: `1e-4`)
- `--actor-lr` (default: `3e-5`)
- `--value-lr` (default: `3e-5`)
- `--collect-interval` (default: `100`)
- `--collect-steps` (default: `1000`)

## Environment Notes

- Requires: `torch`, `gymnasium`, `ale-py`, `numpy`, `matplotlib`, `imageio`.
- Device auto-select: CUDA -> MPS -> CPU.
- Uses `rgb_array` frames resized to `64x64`.
- Training collection and evaluation use an initial `FIRE` action when the environment exposes one.

## Outputs

Each run writes:

`results/dreamer_v3/run_<timestamp>/`

- `model.pth`
- `training_curve.png`
- `dreamer_v3_agent.gif`
