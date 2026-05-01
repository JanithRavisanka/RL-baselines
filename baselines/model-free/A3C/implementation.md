# A3C Implementation Notes

- `ActorCritic`: shared MLP trunk with policy logits and value head.
- `SharedAdam`: optimizer state is placed in shared memory for worker updates.
- Each worker keeps a local model, synchronizes from the global model, collects up to `--rollout-steps` transitions, computes n-step advantage targets, then applies gradients to the global model.
- The training loop uses no replay buffer, matching the on-policy asynchronous design.

For CartPole vector observations, the Atari CNN from the paper is replaced by an MLP; the algorithmic constraints are the A3C constraints.
