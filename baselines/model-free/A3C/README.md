# A3C

Asynchronous Advantage Actor-Critic baseline for `CartPole-v1`.

## Run

```bash
python baselines/model-free/A3C/a3c.py
```

Key paper constraints implemented:
- multiple asynchronous actor-learners,
- shared policy/value network,
- n-step returns with bootstrap value,
- entropy regularization,
- shared optimizer state.

Outputs are written to `results/a3c/run_<timestamp>/`.
