# LeCun-AMI Breakout Experiment Report

## AMI Conditions

| Condition | Seeds | Reward mean | Reward std | Planning rate | Hours mean | Reward/planning | Reward/hour |
|---|---:|---:|---:|---:|---:|---:|---:|

## Breakout Baseline Availability

| Family | Algorithm | Source | Result dir | Frame/update progress | Last-20 reward | Collect reward sum | Eval reward | Failed |
|---|---|---|---|---:|---:|---:|---:|---:|
| model-free | DQN | scheduler_log | results/dqn/run_20260502_001052 | 1000000/1000000 | 0.3 |  | 0.0 | 0 |
| model-free | DDQN | scheduler_log | results/ddqn/run_20260502_001052 | 1000000/1000000 | 0.4 |  | 0.0 | 0 |
| model-free | PER DDQN | scheduler_log | results/per_ddqn/run_20260502_001052 | 1000000/1000000 | 1.45 |  | 0.0 | 0 |
| model-based | Dreamer V2 | scheduler_log | results/dreamer_v2/run_20260502_034615 | 10100/30000 |  | 0.0 |  | 0 |
| model-based | Dreamer V3 | scheduler_log |  | 2100/5000 |  |  |  | 1 |

## Model-Based Baselines Not Used For Breakout

| Algorithm | Environment | Reason |
|---|---|---|
| Dreamer V1 | dm_control/walker-walk-v0 | Not a Breakout/Atari comparison in this repo. |
| MuZero | CartPole-v1 | Current implementation is low-dimensional CartPole, not Breakout pixels. |
| Learned Dynamics MPC | Pendulum-v1 | Continuous-control Pendulum baseline, not Breakout. |
| Dyna-Q | CliffWalking-v1 | Tabular grid-world baseline, not Breakout. |

## Conclusion Template

Adaptive planning improved: TODO after all seeds complete.

Compute efficiency improved: TODO after comparing reward/hour and planning rate.

Notes: AMI uses structured full-game evaluation logs. Existing DQN/DDQN/PER and Dreamer V2/V3 baselines are summarized from artifacts and scheduler logs, so compare them cautiously unless their scripts are upgraded to emit the same periodic evaluation CSV.

AMI runs included: 0
