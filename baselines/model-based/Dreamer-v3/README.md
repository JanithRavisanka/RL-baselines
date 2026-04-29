# Dreamer V3 Baseline (`dreamer_v3.py`)

PyTorch implementation of a Dreamer V3-style agent for discrete-action visual environments (default: Atari Breakout via Gymnasium/ALE).

## Run

From repository root:

```bash
python baselines/model-based/Dreamer-v3/dreamer_v3.py
```

With custom arguments:

```bash
python baselines/model-based/Dreamer-v3/dreamer_v3.py \
  --env ALE/Breakout-v5 \
  --prefill 50000 \
  --updates 5000 \
  --batch-size 16 \
  --seq-len 64
```

## Arguments

- `--env` (str): Gymnasium environment id (must have discrete actions and `rgb_array` rendering).
- `--prefill` (int): random replay prefill steps before model training.
- `--updates` (int): number of training updates.
- `--batch-size` (int): replay batch size (sequence batches).
- `--seq-len` (int): sequence length used for world-model training.

## Environment Expectations

- Python environment with:
  - `torch`, `numpy`, `matplotlib`, `imageio`
  - `gymnasium`, `ale-py`
- Default config expects Atari registration through ALE and runs with pixel observations.
- Device selection is automatic: CUDA -> MPS -> CPU.
- Input frames are resized to `64x64` RGB and stored in replay as `uint8`.

## Outputs

Each run creates a timestamped directory:

- `results/dreamer_v3/run_<YYYYMMDD_HHMMSS>/`

Files generated:

- `model.pth`: trained model weights.
- `training_curve.png`: plot of world/actor/value losses over updates.
- `dreamer_v3_agent.gif`: evaluation rollout rendered as GIF (if frames were captured).

Console logs include prefill progress, periodic loss reports, save paths, and evaluation reward.
