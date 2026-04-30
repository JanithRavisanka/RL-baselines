# Reinforcement Learning Baselines

This repository contains multiple model-free and model-based RL algorithm implementations.

## Algorithms Included

- Model-free:
  - Actor-Critic (`CartPole-v1`, `Pendulum-v1`)
  - DQN (`ALE/Breakout-v5`)
  - Double DQN (`ALE/Breakout-v5`)
  - PER DDQN (`ALE/Breakout-v5`)
- Model-based:
  - Dyna-Q (`CliffWalking-v1`)
  - Learned Dynamics MPC (`Pendulum-v1`)
  - MuZero-style implementation (`CartPole-v1`)
  - Dreamer V1 (`dm_control/walker-walk-v0`)
  - Dreamer V2 (`ALE/Breakout-v5`)
  - Dreamer V3 (`ALE/Breakout-v5`)

## Setup

From repo root:

```bash
cd /home/administrator/baselines
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Atari algorithms (`DQN`, `DDQN`, `PER`) need Atari support:

```bash
python -m pip install "gymnasium[atari]" ale-py
```

## How To Run

Run commands from repo root (`/home/administrator/baselines`).

### Model-Free

- Actor-Critic (CartPole):
```bash
python baselines/model-free/actor-critic/actor_critic.py
```

- Actor-Critic Continuous (Pendulum):
```bash
python baselines/model-free/actor-critic/actor_critic_continuous.py
```

- DQN (Breakout):
```bash
python baselines/model-free/DQN/dqn.py
```

- Double DQN (Breakout):
```bash
python baselines/model-free/DDQN/double_dqn.py
```

- PER DDQN (Breakout):
```bash
python baselines/model-free/PER/per_ddqn.py
```

### Model-Based

- Dyna-Q (CliffWalking):
```bash
python baselines/model-based/Dyna-Q/dyna_q.py
```

- Learned Dynamics MPC (Pendulum):
```bash
python baselines/model-based/MPC/learned_dynamics_mpc.py
```

- MuZero (CartPole):
```bash
python baselines/model-based/MuZero/muzero.py
```

- Dreamer V1 (compact paper-oriented continuous RSSM):
```bash
python baselines/model-based/Dreamer-v1/dreamer_v1.py --env dm_control/walker-walk-v0
```

- Dreamer V2 (compact paper-oriented discrete RSSM):
```bash
python baselines/model-based/Dreamer-v2/dreamer_v2.py --env ALE/Breakout-v5
```

- Dreamer V3 (compact paper-oriented world model + continuation head):
```bash
python baselines/model-based/Dreamer-v3/dreamer_v3.py --env ALE/Breakout-v5
```

### Dreamer Dependencies (target environments)

- For DeepMind Control experiments (Dreamer V1/V2):
```bash
python -m pip install "gymnasium[all]" dm-control shimmy
```

- For Atari experiments (Dreamer V2/V3):
```bash
python -m pip install "gymnasium[atari]" ale-py
```

## Outputs

Each script writes outputs under:

```bash
results/<algorithm_name>/run_<timestamp>/
```

Typical outputs:

- model checkpoint (`.pth` or `.npy`)
- training curve (`.png`)
- evaluation rollout (`.gif`)

## Play Trained Models

Use the unified playback CLI from repo root:

```bash
python play_model.py --algo <algorithm_key>
```

Supported `--algo` values:

- `actor_critic`
- `actor_critic_continuous`
- `dqn`
- `ddqn`
- `per_ddqn`
- `dyna_q`
- `mpc`
- `muzero`

### Common examples

- Auto-pick latest available checkpoint for DDQN:
```bash
python play_model.py --algo ddqn --episodes 1
```

- Use an explicit checkpoint path:
```bash
python play_model.py --algo dqn --model-path results/dqn/run_20260428_081628/model.pth
```

- Save a GIF rollout:
```bash
python play_model.py --algo actor_critic --save-gif results/actor_critic_eval.gif
```

### Notes

- If `--model-path` is omitted, the script searches `results/<algo_folder>/run_*` and picks the latest run containing the expected model file.
- Dyna-Q playback expects a Q-table file (`q_table.npy`).
- `--device` can be `auto`, `cpu`, or `cuda` (default: `auto`).

## Run All Algorithms (Max 3 At Once)

Use the scheduler script to automate running all training scripts with bounded concurrency.

```bash
python run_all_algorithms.py --max-parallel 3
```

Optional flags:

- `--poll-seconds 5` to control completion check interval

Logs are saved to:

```bash
results/scheduler_runs/run_<timestamp>/*.log
```
