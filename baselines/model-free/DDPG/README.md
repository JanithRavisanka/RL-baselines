# DDPG

Deep Deterministic Policy Gradient baseline for `Pendulum-v1`.

## Run

```bash
python baselines/model-free/DDPG/ddpg.py
```

Key paper constraints implemented:
- deterministic actor and Q critic,
- replay buffer,
- target actor/critic networks,
- soft target updates,
- Ornstein-Uhlenbeck exploration noise,
- 400-300 actor/critic MLP structure.

Outputs are written to `results/ddpg/run_<timestamp>/`.
