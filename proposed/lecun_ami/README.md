# LeCun-AMI Proposed Method

This folder contains a practical prototype inspired by Yann LeCun's autonomous
machine intelligence architecture.

The implementation is not a full AGI architecture. It is a bounded RL research
prototype that maps the proposal to implementable modules:

- perception encoder,
- JEPA-style latent predictive world model,
- cost model,
- critic,
- actor,
- configurator-controlled planner.

## Run

From the repository root:

```bash
python proposed/lecun_ami/lecun_ami.py
```

The default environment is `Pendulum-v1`, because cost-guided planning is a
natural fit for continuous control and can be compared against DDPG, TD3, SAC,
and the existing learned-dynamics MPC baseline.

## Complex Game Prototype

For a harder pixel-based game, use the Atari/Breakout variant:

```bash
python proposed/lecun_ami/lecun_ami_atari.py --env ALE/Breakout-v5
```

This version keeps the same AMI structure, but replaces the Pendulum MLP encoder
with a DQN-style CNN encoder, uses a discrete-action latent world model, and
plans over sampled action sequences.

Use this first to verify the server setup:

```bash
python proposed/lecun_ami/lecun_ami_atari.py \
  --env ALE/Breakout-v5 \
  --total-steps 1000 \
  --seed-steps 200 \
  --initial-updates 20 \
  --batch-size 16 \
  --num-sequences 32 \
  --horizon 4 \
  --no-plots \
  --no-gif
```

A more useful comparison run is:

```bash
python proposed/lecun_ami/lecun_ami_atari.py \
  --env ALE/Breakout-v5 \
  --total-steps 500000 \
  --seed-steps 10000 \
  --initial-updates 2000 \
  --batch-size 32 \
  --planning-interval 4 \
  --uncertainty-threshold 0.001 \
  --num-sequences 512 \
  --horizon 12 \
  --critic-n-step 5 \
  --eval-mode adaptive \
  --eval-interval 50000 \
  --eval-episodes 5 \
  --checkpoint-interval 100000
```

Run this separately from `run_all_algorithms.py`; Atari planning is expensive
and should not be mixed into the baseline scheduler unless you intentionally
reserve GPU and CPU budget for it.

## Quick Smoke Test

```bash
python proposed/lecun_ami/lecun_ami.py \
  --episodes 1 \
  --seed-steps 64 \
  --initial-updates 1 \
  --batch-size 32 \
  --num-sequences 8 \
  --horizon 3 \
  --cem-iterations 1 \
  --max-episode-steps 20 \
  --no-plots \
  --no-gif
```

## Main Outputs

Each run writes to:

```text
results/lecun_ami/run_<timestamp>/
```

with:

- `model.pth`,
- `training_curve.png`,
- `planning_rate.png`,
- `lecun_ami_pendulum_agent.gif` unless `--no-gif` is used.

The Atari variant writes to:

```text
results/lecun_ami_atari/run_<timestamp>/
```

with a model checkpoint, `metrics.csv`, optional training plots, and an optional
Atari rollout GIF.

The Atari script also writes research-oriented experiment files:

- `config.json`: exact command configuration and device,
- `training_log.csv`: losses, epsilon, replay size, uncertainty, and rolling rewards,
- `eval_metrics.csv`: periodic full-game evaluation with `terminal_on_life_loss=False`,
- `final_eval_metrics.json`: final full-game evaluation summary,
- `final_summary.json`: compact training summary,
- `checkpoint_step_<step>.pth`: periodic checkpoints when `--checkpoint-interval > 0`,
- `model.pth`: final checkpoint.

## Important Controls

- `--planning-mode adaptive`: configurator chooses planner or actor.
- `--planning-mode always`: always uses cost-guided latent MPC.
- `--planning-mode actor`: uses only the fast actor.
- `--eval-mode planner`: evaluates with the planner.
- `--uncertainty-threshold`: controls adaptive planning sensitivity.
- `--planning-interval`: forces periodic planning even when uncertainty is low.

For Atari, the important additional controls are:

- `--num-sequences`: candidate discrete action sequences per planning step.
- `--horizon`: latent rollout length for each candidate sequence.
- `--actor-seed-frac`: fraction of sequences sampled near the actor policy.
- `--sequence-mutation-prob`: random mutations applied to actor-seeded plans.
- `--critic-n-step`: n-step cost target for the critic; useful for sparse rewards.
- `--critic-target-tau`: EMA rate for the target critic.
- `--eval-interval`: periodic full-game evaluation interval in agent steps.
- `--checkpoint-interval`: periodic checkpoint interval in agent steps.
- `--no-final-eval`: skip the final full-game evaluation for smoke tests.
