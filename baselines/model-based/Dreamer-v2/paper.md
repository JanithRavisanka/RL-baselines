# Dreamer V2 (Paper) - Concise Architecture Notes

Dreamer V2 is a latent world-model RL algorithm designed to learn Atari control directly from pixels by planning in imagination rather than in the real environment.

## 1) Discrete RSSM world model
- Dreamer V2 replaces Dreamer V1's Gaussian latent with a **discrete stochastic latent** (multiple categorical variables) plus a deterministic recurrent state.
- The recurrent state (`h_t`) captures long-term context; the discrete state (`z_t`) captures compact, high-level uncertainty.
- Training uses:
  - a **posterior** \(q(z_t | h_t, e_t)\) that sees the encoded frame,
  - a **prior** \(p(z_t | h_t)\) that predicts without seeing the frame.
- This prior is what enables model rollouts in latent space during policy/value learning.

## 2) Atari focus
- Dreamer V2 was built to be strong on **Atari from raw pixels**.
- Typical Atari setup uses image encoder/decoder, frame preprocessing, and discrete action policies.
- The core idea: fit world dynamics + reward model, then optimize behavior almost entirely from imagined trajectories.

## 3) KL balancing
- Dreamer V2 introduces **KL balancing** (asymmetric gradient flow):
  - one term mainly trains the prior to match posterior statistics (dynamics learning),
  - another term lightly regularizes the posterior toward prior support (representation alignment).
- A weighting (often around `alpha=0.8`) emphasizes predictive prior quality while avoiding posterior collapse.
- Combined with **free nats**, small KL mismatch is tolerated early so useful latent information is preserved.

## 4) Imagined behavior learning
- After world-model updates, Dreamer V2 rolls forward the latent prior for a short horizon.
- The actor samples discrete actions and is optimized from imagined returns (with entropy regularization).
- The value function regresses to lambda-return targets computed on those same imagined rollouts.
- This makes policy improvement data-efficient because expensive real-environment interaction is reduced.

