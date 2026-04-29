# Dreamer V3 Implementation Notes (`dreamer_v3.py`)

This document explains how this specific implementation works and which shared modules it uses from `dreamer_common.py`.

## Model Pieces Used

- `ConvEncoder` / `ConvDecoder`: pixel encoder-decoder for 64x64 RGB frames.
- `DiscreteRSSM`: latent dynamics with:
  - deterministic GRU state (`deter`)
  - stochastic factored categorical state (`stoch` x `classes`)
- `MLPHead` for reward, continuation, value, and actor heads.
- Shared utilities: `symlog`, `symexp`, `lambda_return`, `free_nats_loss`.

In this file, the latent feature for heads is:
- `feat = concat(deter, flatten(stoch))`

## Discrete RSSM and Unimix Usage

- RSSM posterior/prior probabilities are produced every sequence step (`observe_step`) and for imagined steps (`imagine_step`).
- This script applies **unimix** (`mix=0.01`) to categorical probabilities:
  - during world-model sequence updates (posterior and prior probs),
  - and during imagination transitions.
- Unimix mixes a small uniform mass into each categorical distribution to avoid overconfident one-hot collapse and improve training stability.

## Symlog / Symexp Handling

- Reward model is trained in symlog space:
  - target is `symlog(reward)` from replay.
  - loss is MSE between predicted reward head output and symlog target.
- During imagined rollouts, predicted rewards are converted back to environment scale with `symexp` before computing lambda returns.

This keeps regression numerically stable while preserving signed reward information.

## World-Model Update Path

Each update samples replay sequences and performs:

1. Encode observations.
2. RSSM filtering (`observe_step`) with done masking to prevent latent carry-over across episode boundaries.
3. Compute losses:
   - reconstruction loss
   - reward loss (symlog regression)
   - continuation BCE loss
   - KL split via `DiscreteRSSM.kl_balanced`:
     - `dyn_loss` and `rep_loss`, each with free-nats clipping
4. Optimize encoder + RSSM + decoder + reward/continuation heads.

The implemented world loss is:
- `obs + reward + cont + 0.5 * dyn + 0.1 * rep`

## Imagined Behavior Updates

Behavior learning starts from the latest posterior state from replay and rolls out in latent space (`imagine_behavior`):

- Actor samples categorical actions.
- One-hot actions use straight-through gradients.
- RSSM prior transition predicts next imagined latent.
- Continuation head produces per-step discount (`sigmoid(cont) * gamma`).
- Value targets are computed with `lambda_return`.

Then:
- **Return normalizer** updates 5th/95th running quantile bounds on imagined returns.
- Advantage is scaled by robust return range (`high - low`).
- **Actor loss** uses scaled advantage + entropy bonus.
- **Value loss** regresses value head toward imagined lambda-return targets.

## Replay and Training Flow

- Prefill phase gathers random transitions before training.
- Main training alternates:
  - world-model update from replay
  - actor/value updates from imagination
- The script stores model checkpoint and plots losses after training, then runs greedy evaluation to produce a GIF.
