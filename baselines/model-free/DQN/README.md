# DQN Baseline (Breakout)

This folder contains a PyTorch Deep Q-Network baseline for Atari Breakout using replay buffer training, epsilon-greedy exploration, and a periodically synced target network.

## Requirements

- Python 3.10+ recommended
- OS with Gymnasium + ALE support (Linux/macOS tested in most setups)
- Optional but recommended GPU backend:
  - CUDA (NVIDIA) or
  - MPS (Apple Silicon)

Install dependencies (from repo root):

```bash
pip install torch gymnasium ale-py numpy matplotlib imageio
```

If Atari ROM setup is needed in your environment, install Gymnasium Atari extras:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```

## Run Training

From this directory:

```bash
python dqn.py
```

## Inputs / Arguments

- Script currently has no CLI flags.
- Core defaults are defined in code:
  - env: `ALE/Breakout-v5`
  - `max_frames=1000000`
  - `batch_size=32`
  - `gamma=0.99`
  - optimizer: DeepMind-style RMSProp (`lr=2.5e-4`, `alpha=0.95`, `momentum=0.95`, `eps=0.01`)
  - replay warmup: `50000` transitions
  - epsilon schedule: `1.0 -> 0.1` over `250000` agent decisions (`1M` Atari frames with `frame_skip=4`)
  - target update frequency: every `10000` frames

The script will:

1. create a timestamped output directory under `results/dqn/`,
2. train for `1000000` frames,
3. save the model checkpoint,
4. save a training curve plot,
5. run one evaluation episode and save a GIF.

## Outputs Generated

For a run timestamp `run_YYYYMMDD_HHMMSS`, outputs are:

- `results/dqn/run_.../model.pth`
- `results/dqn/run_.../training_curve.png`
- `results/dqn/run_.../breakout_dqn_agent.gif`

During training, console logs appear every 2000 frames:

- frame progress,
- current epsilon,
- average reward over the last 20 completed episodes.

## Notes

- Input preprocessing uses grayscale 84x84 frames with 4-frame stacking.
- Pixel normalization to `[0,1]` is done in tensor conversion paths (`/255.0`).
- Rewards are clipped to `{-1, 0, +1}` for optimization stability.
