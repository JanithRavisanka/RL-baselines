# Prioritized Experience Replay with Double DQN

Prioritized Experience Replay (PER) and Double DQN (DDQN) are complementary:

- **DDQN** reduces Q-value overestimation by selecting the next action with the online network and evaluating it with the target network.
- **PER** improves sample efficiency by replaying transitions with larger temporal-difference (TD) errors more often.

In practice, DDQN stabilizes targets, while PER focuses updates on transitions that currently carry higher learning signal.

## Core idea

Instead of sampling uniformly from replay memory, PER samples transition `i` with probability:

\[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
\]

where:

- `p_i` is the transition priority (typically based on TD error),
- `alpha` controls prioritization strength (`0` = uniform replay, `1` = full prioritization).

## SumTree data structure

A SumTree stores priorities in leaf nodes and cumulative sums in internal nodes:

- Root node stores the total priority mass.
- Sampling draws a value `s` in `[0, total_priority)` and traverses the tree by comparing `s` against left-subtree sums.
- Insert/update and sample are both `O(log N)`.

This enables efficient priority-based replay at scale.

## Importance Sampling (IS) correction

Because PER changes the sampling distribution, updates become biased unless corrected. IS reweights each sampled transition:

\[
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
\]

- `beta` anneals toward `1.0` over training,
- weights are usually normalized by dividing by `max(w_i)` in a batch.

Applying `w_i` to per-sample loss reduces the bias introduced by non-uniform sampling, while still benefiting from PER's efficiency.
