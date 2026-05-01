# Double DQN Baseline (Breakout)

This folder contains a concise Double DQN Atari baseline implementation in `double_dqn.py`.

## Run

From the repo root:

```bash
python baselines/model-free/DDQN/double_dqn.py
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

## What the script does

- Trains a Double DQN agent on `ALE/Breakout-v5` for up to 1M frames
- Uses replay buffer, target network updates, epsilon decay, reward clipping, and Huber loss
- Evaluates one greedy episode after training and exports a GIF

## Outputs

Each run creates a timestamped directory:

- `results/ddqn/run_<timestamp>/model.pth` - trained weights
- `results/ddqn/run_<timestamp>/training_curve.png` - episode reward curve (with moving average)
- `results/ddqn/run_<timestamp>/breakout_ddqn_agent.gif` - evaluation rollout

During training, the script logs frame count, epsilon, and recent average reward.

## Notes

- First run may require Atari dependencies (`gymnasium[atari]`, `ale-py`, PyTorch, matplotlib, imageio).
- Training is compute-intensive; GPU/MPS is used automatically when available.
