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
- This script sets RSSM **unimix** (`mix=0.01`) before categorical sampling:
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

1. Infer the initial posterior from the first current observation in the chunk.
2. Apply each action and condition the posterior on the true next observation for that transition.
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

Behavior learning samples nonterminal posterior states from across the replay sequence and rolls out in latent space (`imagine_behavior`):

- Actor samples categorical actions.
- Sampled one-hot actions are fed to the fixed RSSM rollout.
- RSSM prior transition predicts next imagined latent.
- Continuation head produces per-step discount (`sigmoid(cont) * gamma`).
- Value targets are computed with `lambda_return`.

Then:
- **Return normalizer** updates 5th/95th running quantile bounds on imagined returns.
- Advantage is scaled by robust return range (`high - low`).
- **Actor loss** uses a REINFORCE-style scaled advantage + entropy bonus for discrete actions.
- **Value loss** regresses value head toward imagined lambda-return targets.

## Replay and Training Flow

- Prefill phase gathers random transitions before training.
- Replay stores aligned `(obs, action, reward, done, next_obs)` transitions.
- Sequence sampling rejects chunks that cross episode boundaries before the final transition, so recurrent state is not carried between unrelated episodes.
- Main training alternates:
  - world-model update from replay
  - actor/value updates from imagination
- Every `--collect-interval` updates, the current actor collects `--collect-steps` new environment steps into replay.
- The script stores model checkpoint and plots losses after training, then runs greedy evaluation to produce a GIF.
