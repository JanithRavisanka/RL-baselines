# Reinforcement Learning Baselines

This repository contains multiple model-free and model-based RL algorithm implementations.

## Algorithms Included

- Model-free:
  - Actor-Critic (`CartPole-v1`, `Pendulum-v1`)
  - DQN (`ALE/Breakout-v5`)
  - Double DQN (`ALE/Breakout-v5`)
  - PER DDQN (`ALE/Breakout-v5`)
- Model-based:
  - Dyna-Q (`CliffWalking-v0`)
  - Learned Dynamics MPC (`Pendulum-v1`)
  - MuZero-style implementation (`CartPole-v1`)

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

## Outputs

Each script writes outputs under:

```bash
results/<algorithm_name>/run_<timestamp>/
```

Typical outputs:

- model checkpoint (`.pth` or `.npy`)
- training curve (`.png`)
- evaluation rollout (`.gif`)
