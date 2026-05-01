# PPO Implementation Notes

- `PPOActorCritic`: Tanh MLP with policy logits and value head.
- Collects fixed-length on-policy rollouts.
- Computes GAE advantages and return targets.
- Optimizes the clipped surrogate objective:
  - `min(ratio * advantage, clipped_ratio * advantage)`.
- Runs multiple minibatch epochs per rollout, as proposed in PPO.

The implementation uses discrete CartPole actions; continuous-action PPO would use a Gaussian policy with the same clipped objective.
