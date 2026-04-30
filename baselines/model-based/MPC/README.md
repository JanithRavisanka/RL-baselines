# Learned Dynamics MPC (Pendulum-v1)

This folder contains a model-based RL baseline that learns environment dynamics and performs online MPC planning with CEM.

## Run

From repo root:

```bash
python baselines/model-based/MPC/learned_dynamics_mpc.py
```

The script auto-selects device (`cuda`, `mps`, or `cpu`) and creates a timestamped results directory.

## Inputs / Arguments

```bash
python baselines/model-based/MPC/learned_dynamics_mpc.py \
  --ensemble-size 7 \
  --seed-steps 10000 \
  --initial-epochs 300 \
  --retrain-epochs 30 \
  --num-episodes 100 \
  --batch-size 512 \
  --num-sequences 2048 \
  --horizon 30 \
  --cem-iterations 6 \
  --elite-frac 0.05 \
  --validation-horizon 10 \
  --gamma 0.99 \
  --lr 1e-3
```

Defaults:
- `--ensemble-size 7`
- `--seed-steps 10000`
- `--initial-epochs 300`
- `--retrain-epochs 30`
- `--num-episodes 100`
- `--batch-size 512`
- `--num-sequences 2048`
- `--horizon 30`
- `--cem-iterations 6`
- `--elite-frac 0.05`
- `--validation-horizon 10`
- `--gamma 0.99`
- `--lr 1e-3`

## Outputs

For each run, files are saved under:

- `results/mpc/run_<timestamp>/`

Expected artifacts:

- `dynamics_ensemble.pth`  
  Saved ensemble model `state_dict`s, dynamics normalizer statistics, and planner config.
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

- The planner is receding-horizon MPC: it plans over `--horizon` steps but executes only the first action, then replans next step.
- Default research preset uses horizon 30 (`--horizon 30`).
- Dynamics are learned in delta-space with normalization for more stable optimization.
