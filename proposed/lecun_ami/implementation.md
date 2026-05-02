# Implementation Mapping

This implementation is structured around LeCun's proposed autonomous machine
intelligence components, but kept small enough for FYP experiments.

## 1) Perception Encoder

`Encoder` maps the raw Gymnasium state into a compact latent vector:

```text
s_t -> z_t
```

For `Pendulum-v1`, this is an MLP. For image environments, this component can be
replaced by a CNN or ViT-style encoder.

The Atari extension in `lecun_ami_atari.py` uses a DQN-style CNN encoder:

```text
4 x 84 x 84 stacked grayscale frames -> z_t
```

This keeps the representation comparable to DQN/DDQN/PER baselines while still
feeding a latent world model instead of a direct Q-value head.

## 2) JEPA-Style World Model

`LatentWorldModel` predicts the next latent representation:

```text
world_model(z_t, a_t) -> z_{t+1}
```

The target is produced by an EMA target encoder:

```text
target_encoder(s_{t+1}) -> stop_gradient(z_{t+1})
```

The model predicts in latent space instead of reconstructing pixels.

For Atari, `DiscreteLatentWorldModel` conditions the latent transition on an
embedded discrete action:

```text
world_model(z_t, embed(a_t)) -> z_{t+1}
```

An ensemble is used so the configurator can estimate model uncertainty from
disagreement between predicted next latents.

## 3) Cost Module

`CostModel` predicts immediate cost:

```text
cost = -reward
```

This keeps the implementation compatible with standard Gymnasium RL tasks while
leaving room to add safety, uncertainty, or task-constraint costs later.

## 4) Critic

`Critic` estimates long-term cost-to-go from the latent state:

```text
V(z_t) ~= expected future cost
```

The planner uses this as a terminal value after the finite CEM rollout horizon.

## 5) Actor

`Actor` is the fast reactive policy. It is trained by imitating actions selected
by the planner. This corresponds to distilling deliberate planning into fast
action selection.

For Atari, `DiscreteActor` outputs action logits and is trained with
cross-entropy on planner-selected actions stored in replay.

## 6) Configurator

The configurator is implemented in `should_plan()`. It decides whether to use:

- the fast actor, or
- cost-guided latent planning.

The default rule triggers planning during warmup, periodically, or when ensemble
world-model uncertainty exceeds a threshold.

## 7) Planner

The planner uses Cross-Entropy Method MPC in latent space. It samples candidate
action sequences, rolls them forward through the world-model ensemble, scores
them using predicted cost plus uncertainty penalty, and executes the first action
of the best sequence.

The Atari extension uses discrete random shooting instead of continuous CEM:

```text
sample action sequences -> rollout in latent space -> choose lowest-cost sequence
```

Some candidate sequences are sampled from the actor distribution and mutated, so
planning can search near the current policy while still testing random plans.

## Scope

This is a concrete research prototype, not a claim of complete autonomous
machine intelligence. It implements the subset needed for empirical comparison:

```text
latent world model + cost model + actor + critic + configurator + bounded planner
```

## Complex-Game Evaluation

Use `lecun_ami_atari.py` for the first complex-game comparison. A fair initial
study is to compare it against the existing Atari baselines on the same
environment:

```text
ALE/Breakout-v5: DQN, DDQN, PER DDQN, LeCun-AMI Atari
```

Track episode reward, planning rate, wall-clock time, and GPU memory. The
planning rate is important because this method explicitly trades extra
deliberation for better action selection.
