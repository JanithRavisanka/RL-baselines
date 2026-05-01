# TD3

Twin Delayed DDPG baseline for `Pendulum-v1`.

## Run

```bash
python baselines/model-free/TD3/td3.py
```

Key paper constraints implemented:
- clipped double Q-learning with twin critics,
- delayed actor and target updates,
- target policy smoothing,
- DDPG-style deterministic actor.

Outputs are written to `results/td3/run_<timestamp>/`.
