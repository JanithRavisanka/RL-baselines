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

The script will:

1. create a timestamped output directory under `results/dqn/`,
2. train for `500000` frames,
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
