# Learned Dynamics MPC (Pendulum-v1)

This folder contains a model-based RL baseline that learns environment dynamics and performs online MPC planning with CEM.

## Run

From repo root:

```bash
python baselines/model-based/MPC/learned_dynamics_mpc.py
```

The script auto-selects device (`cuda`, `mps`, or `cpu`) and creates a timestamped results directory.

## Inputs and settings (in code)

Environment and core settings are defined directly in `learned_dynamics_mpc.py`:

- Environment: `gym.make("Pendulum-v1")`
- Dynamics ensemble size: `5`
- Replay buffer capacity: `100000`
- Seed random transitions: `2000`
- Initial dynamics training: `epochs=100`, `batch_size=256`
- Online MPC episodes: `25`
- Per-episode retraining: `epochs=10`, `batch_size=256`
- CEM planner:
  - `num_sequences=512` (training), `1024` (evaluation)
  - `horizon=20`
  - `elite_frac=0.1`
  - `iterations=4`
  - `gamma=0.99`

No separate config file is required for default execution.

## Outputs

For each run, files are saved under:

- `results/mpc/run_<timestamp>/`

Expected artifacts:

- `dynamics_ensemble.pth`  
  Saved list of 5 model `state_dict`s.
- `training_curve.png`  
  Episode reward curve for the online MPC phase.
- `pendulum_mpc_agent.gif`  
  Rendered evaluation rollout with the trained planner.

## Logged metrics and meaning

During training, each episode prints:

- `Total Reward`  
  Sum of environment rewards in that episode.  
  For `Pendulum-v1`, values closer to `0` are better (less negative is better).
- `One-step MSE`  
  Mean supervised loss on normalized delta prediction during dynamics retraining.
- `5-step rollout MSE`  
  Multistep validation error from imagined 5-step rollouts in replay transitions.  
  This reflects compounding model error and is often more relevant for planning quality than one-step loss alone.

## Notes

- The planner is receding-horizon MPC: it plans over 20 steps but executes only the first action, then replans next step.
- Dynamics are learned in delta-space with normalization for more stable optimization.
