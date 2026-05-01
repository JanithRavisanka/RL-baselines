# DDPG Implementation Notes

- Actor: 400 -> 300 ReLU MLP with tanh-scaled deterministic action output.
- Critic: state pathway first, action concatenated at the second hidden layer.
- Exploration uses an Ornstein-Uhlenbeck process.
- Critic target:
  - `r + gamma * Q_target(s', actor_target(s'))`
- Actor objective:
  - maximize critic value for current actor actions.
- Target networks use Polyak averaging with `--tau`.

This is a low-dimensional Pendulum version of the DDPG control pipeline, not the paper's pixel-input variant.
