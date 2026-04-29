# Actor-Critic Baselines (Discrete + Continuous)

This folder provides two PyTorch actor-critic examples:
- `actor_critic.py` for `CartPole-v1` (discrete actions)
- `actor_critic_continuous.py` for `Pendulum-v1` (continuous actions)

## Run

From this directory:

```bash
python actor_critic.py
python actor_critic_continuous.py
```

If you use a virtual environment, activate it first (example):

```bash
source venv/bin/activate
```

## Inputs / Arguments

- Both scripts currently take **no CLI arguments**.
- Key settings are hard-coded defaults in each script:
  - `actor_critic.py`: `gamma=0.99`, `lr=0.01`, `num_episodes=1000`
  - `actor_critic_continuous.py`: `gamma=0.6`, `lr=0.001`, `num_episodes=100000`
- Continuous script auto-selects device in order: CUDA, MPS, then CPU.

## Outputs

Both scripts:
- print training progress every 50 episodes
- save a trained model checkpoint `model.pth`
- save a reward curve PNG
- run deterministic evaluation and save a GIF rollout

Discrete script (`actor_critic.py`) writes:
- `training_curve.png`
- `cartpole_agent.gif`

Continuous script (`actor_critic_continuous.py`) writes:
- `training_curve_continuous.png`
- `pendulum_agent.gif`

## Save Paths

Each run creates a timestamped results directory under the repo root:

- `actor_critic.py`:
  - `results/actor_critic/run_YYYYMMDD_HHMMSS/`
- `actor_critic_continuous.py`:
  - `results/actor_critic_continuous/run_YYYYMMDD_HHMMSS/`

All artifacts for that run are stored inside its run folder.
