# Reinforcement Learning Baselines

This repository contains multiple model-free and model-based RL algorithm implementations.

## Algorithms Included

- Model-free:
  - A3C (`CartPole-v1`)
  - PPO (`CartPole-v1`)
  - DDPG (`Pendulum-v1`)
  - TD3 (`Pendulum-v1`)
  - SAC (`Pendulum-v1`)
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

- A3C (CartPole):
```bash
python baselines/model-free/A3C/a3c.py
```

- PPO (CartPole):
```bash
python baselines/model-free/PPO/ppo.py
```

- DDPG (Pendulum):
```bash
python baselines/model-free/DDPG/ddpg.py
```

- TD3 (Pendulum):
```bash
python baselines/model-free/TD3/td3.py
```

- SAC (Pendulum):
```bash
python baselines/model-free/SAC/sac.py
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

- `a3c`
- `ppo`
- `ddpg`
- `td3`
- `sac`
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
python play_model.py --algo ppo --save-gif results/ppo_eval.gif
```

### Notes

- If `--model-path` is omitted, the script searches `results/<algo_folder>/run_*` and picks the latest run containing the expected model file.
- Dyna-Q playback expects a Q-table file (`q_table.npy`).
- `--device` can be `auto`, `cpu`, or `cuda` (default: `auto`).

## Run All Algorithms

Use the resource-aware scheduler to run all training scripts in parallel. Instead of a
flat concurrency cap, it tracks GPU VRAM and CPU slot budgets so lightweight jobs never
block behind heavy GPU workloads.

```bash
# Smart defaults — auto-skips already-completed algorithms
python run_all_algorithms.py

# Preview the scheduling plan without launching anything
python run_all_algorithms.py --dry-run

# Force re-run all algorithms (ignore existing results)
python run_all_algorithms.py --no-skip

# Run only specific algorithms
python run_all_algorithms.py --only dqn muzero dreamer_v3

# Exclude slow algorithms
python run_all_algorithms.py --exclude dreamer_v3

# Override GPU budget (e.g., if other processes are using the GPU)
python run_all_algorithms.py --gpu-budget-mb 10000

# Legacy flat-parallelism mode (backward compatible)
python run_all_algorithms.py --max-parallel 3
```

### Optimized Server Run

For the RTX PRO 4000 / 24-core server, use the staged launcher. It does not change
any learning algorithm; it only caps CPU thread oversubscription and runs compatible
jobs together.

```bash
python run_optimized_server.py
```

Useful variants:

```bash
# Preview the staged plan
python run_optimized_server.py --dry-run

# Retrain even if previous outputs exist
python run_optimized_server.py --force

# Run selected stages
python run_optimized_server.py --stage atari_value
python run_optimized_server.py --stage dreamer_v2
```

Stages:

- `light`: `dyna_q ppo ddpg td3 sac mpc`
- `atari_value`: `dqn ddqn per_ddqn`
- `muzero`: `muzero`
- `dreamer_v1`: `dreamer_v1`
- `dreamer_v2`: `dreamer_v2`
- `dreamer_v3`: `dreamer_v3`

### Scheduler flags

| Flag | Default | Description |
|---|---|---|
| `--gpu-budget-mb` | `14000` | Total GPU VRAM budget in MB |
| `--cpu-slots` | `10` | Max CPU weight slots to use concurrently |
| `--poll-seconds` | `10` | Job completion polling interval |
| `--no-skip` | off | Force re-run even if results exist |
| `--dry-run` | off | Print scheduling plan without running |
| `--only ALGO [...]` | all | Run only the listed algorithm(s) |
| `--exclude ALGO [...]` | none | Skip the listed algorithm(s) |
| `--max-parallel N` | — | Legacy mode: flat slot cap, ignores budgets |

### How it works

Each algorithm is annotated with its GPU VRAM estimate and CPU weight. The scheduler
launches jobs whenever the remaining budget allows, preferring shortest-estimated-duration
jobs first. CPU-only algorithms (e.g., `a3c`, `dyna_q`) run freely alongside
GPU jobs without consuming GPU budget. Already-completed algorithms are auto-detected
and skipped.

Logs are saved to:

```bash
results/scheduler_runs/run_<timestamp>/*.log
```
