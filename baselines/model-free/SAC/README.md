# SAC

Soft Actor-Critic baseline for `Pendulum-v1`.

## Run

```bash
python baselines/model-free/SAC/sac.py
```

Key paper constraints implemented:
- maximum-entropy actor objective,
- tanh-squashed Gaussian policy,
- twin Q-functions,
- entropy temperature optimization,
- off-policy replay buffer.

Outputs are written to `results/sac/run_<timestamp>/`.
