# PPO

Proximal Policy Optimization baseline for `CartPole-v1`.

## Run

```bash
python baselines/model-free/PPO/ppo.py
```

Key paper constraints implemented:
- clipped probability-ratio surrogate objective,
- multiple epochs over each on-policy rollout,
- generalized advantage estimation,
- value loss and entropy bonus.

Outputs are written to `results/ppo/run_<timestamp>/`.
