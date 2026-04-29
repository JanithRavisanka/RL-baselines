# Dreamer V1 (Paper) - Architecture and Training Loop

Dreamer V1 (Hafner et al., 2019) learns control directly from latent imagination instead of learning only from real transitions.

## Core Architecture

- **World model**
  - **Encoder** maps pixel observation \(o_t\) to embedding \(e_t\).
  - **RSSM** keeps:
    - deterministic memory \(h_t\),
    - stochastic latent \(z_t\).
  - Two latent distributions are learned:
    - **prior** \(p(z_t \mid h_t)\) from dynamics only,
    - **posterior** \(q(z_t \mid h_t, e_t)\) conditioned on observation.
  - **Decoder** reconstructs pixels from latent feature \([h_t, z_t]\).
  - **Reward head** predicts \(r_t\) from latent feature.

- **Behavior model**
  - **Actor** outputs continuous actions from latent features.
  - **Value model (critic)** predicts latent state value.

## World Model Objective

World model is trained on replay sequences by maximizing a variational objective (implemented as losses):

- reconstruction loss: make decoded image match observed image,
- reward prediction loss: make predicted reward match true reward,
- KL regularization: keep posterior close to prior,
- free-bits / free-nats threshold to avoid over-regularizing small KL early.

## Imagination-Based Behavior Learning

After fitting the world model on real data:

1. Start from posterior latent states inferred from replay.
2. Roll dynamics forward in latent space using prior transitions and actor actions (no environment interaction).
3. Predict imagined rewards and values along the latent rollout.
4. Compute lambda-returns on imagined trajectories.
5. Update:
   - actor to maximize imagined return (plus small entropy bonus),
   - value model to regress to imagined lambda-return targets.

This is Dreamer’s key idea: policy/value optimize expected return under learned latent dynamics.

## High-Level Training Loop

1. Collect initial random dataset (replay prefill).
2. Repeat:
   - sample sequence batch from replay,
   - train world model on real sequences,
   - imagine latent rollouts from posterior states,
   - update actor and value on imagined returns,
   - periodically evaluate current policy in real environment.
