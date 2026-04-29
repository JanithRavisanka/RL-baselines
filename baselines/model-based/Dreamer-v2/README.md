# Dreamer V2 Baseline

PyTorch Dreamer V2-style Atari training script using a discrete RSSM world model and imagined behavior learning.

## Run

From repo root:

```bash
python baselines/model-based/Dreamer-v2/dreamer_v2.py --env ALE/Breakout-v5 --prefill 20000 --updates 3000
```

## Main arguments
- `--env` (default: `ALE/Breakout-v5`): Gymnasium Atari env id.
- `--prefill` (default: `20000`): random steps collected before gradient updates.
- `--updates` (default: `3000`): number of training updates.

## Atari environment notes
- Uses `gymnasium` + `ale_py` with `render_mode="rgb_array"`.
- Frames are preprocessed to `64x64` and 3 channels.
- Action space is assumed discrete (`env.action_space.n`).
- Evaluation includes a Breakout-specific `FIRE` action after reset to start gameplay.

## Outputs generated
Each run creates a timestamped directory:

- `results/dreamer_v2/run_<timestamp>/model.pth` - trained weights
- `results/dreamer_v2/run_<timestamp>/training_curve.png` - world/actor/value losses
- `results/dreamer_v2/run_<timestamp>/dreamer_v2_agent.gif` - evaluation rollout (if frames captured)

The script also prints periodic training losses and final evaluation reward.

