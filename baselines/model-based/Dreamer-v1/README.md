# Dreamer V1 (Pixel Control)

PyTorch Dreamer V1 baseline using a continuous RSSM and dm_control-style continuous-action environments.

## Run

From repo root:

```bash
python baselines/model-based/Dreamer-v1/dreamer_v1.py \
  --env dm_control/walker-walk-v0 \
  --prefill 5000 \
  --updates 2000
```

Arguments:
- `--env`: Gymnasium environment id (default `dm_control/walker-walk-v0`).
- `--prefill`: random replay prefill steps before training.
- `--updates`: number of gradient update iterations.

## Environment / Dependency Notes

- Requires `gymnasium`, `torch`, `numpy`, `matplotlib`, `imageio`, `shimmy`.
- dm_control tasks are expected through Gymnasium + shimmy registration (`gym.register_envs(shimmy)`).
- The script creates envs with `render_mode="rgb_array"` for pixel observations and GIF recording.
- In headless Linux (no `DISPLAY`), it sets:
  - `MUJOCO_GL=egl`
  - `PYOPENGL_PLATFORM=egl`
- If the env observation is not already image-like, the code falls back to `env.render()` and resizes to `64x64`.

## Outputs

Each run creates:

`results/dreamer_v1/run_<timestamp>/`

with:
- `model.pth` - trained model weights.
- `training_curve.png` - world/actor/value loss curves.
- `dreamer_v1_agent.gif` - evaluation rollout video.

The script prints the full run directory path at startup and confirms each saved artifact path.
