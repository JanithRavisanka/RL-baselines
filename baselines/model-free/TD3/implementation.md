# TD3 Implementation Notes

- Actor: deterministic 400 -> 300 ReLU MLP.
- Critic: two independent Q networks.
- Target uses:
  - target actor action plus clipped Gaussian smoothing noise,
  - minimum of target Q values.
- Actor update is delayed by `--policy-delay`.
- Target networks use Polyak averaging.

These are the three TD3 mechanisms that distinguish it from DDPG.
