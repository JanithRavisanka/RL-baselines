# Dreamer V3 (Paper Notes)

Dreamer V3 is a latent world-model reinforcement learning method designed to use one recipe across many domains (Atari, DMControl, Minecraft-like tasks, and more) with minimal retuning.

## Core Architecture

- **World model**: encodes observations into compact latent states and learns latent dynamics.
- **Behavior learning in imagination**: actor and value are updated mostly from imagined trajectories rolled out in latent space, not from expensive pixel environment interactions.
- **Decoder/reward/continuation heads**: reconstruct observations and predict reward and continuation (non-terminal probability), enabling self-supervised representation plus control signals.

## Robust Scaling Ideas

- **Symlog targets**: reward/value-like targets are transformed with a signed log transform (`symlog`) before regression.  
  This keeps small values near-linear while compressing large magnitudes.
- **Symexp inversion**: when computing returns in environment scale, predictions are mapped back with `symexp`.
- **Return normalization**: policy gradients are scaled using robust running statistics of imagined returns (quantile/range based, not only mean/std), reducing sensitivity to outliers and changing reward scales.

These mechanisms make optimization less brittle across tasks with very different reward distributions.

## KL Split / Balanced KL

Instead of a single symmetric KL regularizer, Dreamer-style training separates pressure into two components:

- **Dynamics KL (`dyn`)**: trains prior dynamics to match posterior statistics inferred from observations.
- **Representation KL (`rep`)**: trains encoder/posterior to stay close to prior support.

Using separate weights and free-bits/free-nats style clipping prevents over-regularization early while keeping latent rollouts coherent for planning/imagination.

## Broad-Domain Recipe

Dreamer V3 emphasizes one high-level recipe:

1. Collect and replay trajectories.
2. Train world model on sequences (reconstruction + reward + continuation + KL regularization).
3. Train actor-critic on imagined latent rollouts.
4. Repeat with stable defaults (optimizer settings, normalization, KL handling) rather than per-domain algorithm changes.

The main claim is not only sample efficiency, but **robustness of one training setup across heterogeneous environments**.
