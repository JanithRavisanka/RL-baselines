# SAC Implementation Notes

- Actor: stochastic Gaussian policy with tanh correction in the log-probability.
- Critics: twin Q networks to reduce overestimation.
- Critic target includes entropy:
  - `r + gamma * (min(Q_target) - alpha * log pi(a'|s'))`
- Actor maximizes Q while retaining entropy:
  - `alpha * log pi(a|s) - min(Q(s,a))`
- `alpha` is optimized toward the target entropy.

This is the modern continuous-control SAC formulation commonly used as the baseline version.
