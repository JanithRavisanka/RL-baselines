# Learned Dynamics MPC (Nagabandi/PETS-style)

This note summarizes the idea behind model-based control with a learned dynamics model and CEM planning, in the spirit of works such as Nagabandi et al. and PETS.

## Core idea

Instead of directly learning a policy, learn a model of environment dynamics:

- Predict state change (delta) with a neural network:
  - `s_{t+1} = s_t + f_theta(s_t, a_t)`
- Use that model online to plan future action sequences.
- Execute only the first action from the best planned sequence, then re-plan at the next real state (MPC / receding horizon).

This gives strong sample efficiency because each real transition is reused to improve the model and all planning happens in imagination.

## Objective

For a planning horizon `H`, pick an action sequence that maximizes predicted return under learned dynamics:

- `max_{a_{t:t+H-1}} sum_{k=0}^{H-1} gamma^k r(s_{t+k+1}, a_{t+k})`
- Subject to model rollout:
  - `s_{t+k+1} = s_{t+k} + f_theta(s_{t+k}, a_{t+k})`

In PETS-like settings, ensembles are used to reduce model bias and improve robustness during planning.

## Planning with CEM

Cross-Entropy Method (CEM) is a derivative-free optimizer over action sequences:

1. Initialize Gaussian over `H x action_dim` action sequence.
2. Sample many candidate sequences.
3. Roll each sequence forward in the learned model and score return.
4. Select elites (top fraction by return).
5. Refit Gaussian to elites.
6. Repeat for a few iterations.
7. Execute first action of the best sequence.

## Typical learning-control cycle

1. **Collect seed data** with random or simple behavior.
2. **Train dynamics model** on replayed transitions.
3. **Run MPC** in the real environment using model rollouts.
4. **Aggregate new transitions** from executed actions.
5. **Retrain/fine-tune model** and repeat.

This closes the loop between model learning and control, continuously correcting model errors in regions visited by the planner.
