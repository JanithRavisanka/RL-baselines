# Breakout Baseline Architecture Audit

This audit checks the Breakout comparison baselines against their original paper-oriented architectures. The standard is paper-oriented rather than strict replica: core algorithmic architecture should match, while resource and environment simplifications are acceptable only when they are documented clearly.

## Primary Sources

- DQN: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Double DQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- PER: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- Dreamer V2: [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
- Dreamer V3: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)

## Executive Findings

| Baseline | Overall status | Main finding |
|---|---|---|
| DQN | Defensible with reporting fixes | Core DQN architecture is present; experiment logging/evaluation is not yet AMI-comparable. |
| Double DQN | Defensible with reporting fixes | Correct Double DQN target path and DQN parity; same reporting gap as DQN. |
| PER DDQN | Defensible with reporting fixes | PER, DDQN target, alpha/beta, and IS-weighted loss are implemented; same reporting gap. |
| Dreamer V2 | Compact paper-oriented baseline | Core discrete-world-model Dreamer architecture is present; Atari protocol differs from DQN/AMI wrappers. |
| Dreamer V3 | Not final-run ready | Core Dreamer V3-style pieces are present, but current logs show a CUDA OOM failure and reporting/protocol gaps. |

## Severity Summary

Critical:

- Dreamer V3 is not final-run ready until a safe memory preset or runtime guard is added.

Major:

- All five baselines need structured config/evaluation outputs for a fair AMI comparison.
- Dreamer V2/V3 use a compact raw-RGB 64x64 preprocessing path rather than the DQN/AMI AtariPreprocessing stack, so their scores are not protocol-identical unless this is documented or standardized.

Documentation:

- DQN/DDQN/PER replay capacity is reduced to `100000`, which is acceptable for the FYP setup but should be stated as a memory-driven simplification.
- Dreamer V3 should be described as compact Dreamer V3-style unless the full official recipe is implemented.

## DQN

Status: defensible as a paper-oriented DQN baseline after reporting fixes.

| Checklist item | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| Atari preprocessing, frame stacking, reward clipping | Match | Minor | Uses `AtariPreprocessing` with frame skip, grayscale, life-loss terminals, and 4-frame stacking; clips rewards with `np.sign`: [dqn.py:134](../../../baselines/model-free/DQN/dqn.py#L134), [dqn.py:146](../../../baselines/model-free/DQN/dqn.py#L146), [dqn.py:153](../../../baselines/model-free/DQN/dqn.py#L153), [dqn.py:227](../../../baselines/model-free/DQN/dqn.py#L227). | Keep. Document training versus evaluation life-loss behavior. |
| DeepMind-style CNN Q-network | Match | Minor | Conv layers 32/64/64 with 8/4/3 kernels and 512-unit hidden head: [dqn.py:72](../../../baselines/model-free/DQN/dqn.py#L72), [dqn.py:82](../../../baselines/model-free/DQN/dqn.py#L82), [dqn.py:88](../../../baselines/model-free/DQN/dqn.py#L88). | Keep. |
| Replay buffer and warmup | Acceptable simplification | Documentation | Uniform replay is present and learning starts at 50000 transitions; capacity is `100000`: [dqn.py:105](../../../baselines/model-free/DQN/dqn.py#L105), [dqn.py:181](../../../baselines/model-free/DQN/dqn.py#L181), [dqn.py:195](../../../baselines/model-free/DQN/dqn.py#L195). | Document replay capacity as memory-driven. |
| Target network hard sync | Match | Minor | Initial copy and periodic hard sync every 10000 frames: [dqn.py:173](../../../baselines/model-free/DQN/dqn.py#L173), [dqn.py:196](../../../baselines/model-free/DQN/dqn.py#L196), [dqn.py:281](../../../baselines/model-free/DQN/dqn.py#L281). | Keep. |
| Epsilon-greedy schedule | Match | Minor | Epsilon decays from 1.0 to 0.1 over 250000 agent decisions, aligned with frame skip 4: [dqn.py:188](../../../baselines/model-free/DQN/dqn.py#L188), [dqn.py:190](../../../baselines/model-free/DQN/dqn.py#L190), [dqn.py:240](../../../baselines/model-free/DQN/dqn.py#L240). | Keep. |
| Bellman target, Huber loss, RMSProp-style optimizer | Match | Minor | Uses DeepMind-style RMSProp, target-network max backup, and smooth L1 loss: [dqn.py:25](../../../baselines/model-free/DQN/dqn.py#L25), [dqn.py:175](../../../baselines/model-free/DQN/dqn.py#L175), [dqn.py:270](../../../baselines/model-free/DQN/dqn.py#L270). | Keep. |
| Evaluation and reproducibility | Missing | Major | Evaluation is a single GIF-style rollout and does not write config/eval CSV/final JSON: [dqn.py:313](../../../baselines/model-free/DQN/dqn.py#L313), [dqn.py:318](../../../baselines/model-free/DQN/dqn.py#L318), [dqn.py:382](../../../baselines/model-free/DQN/dqn.py#L382). | Add seed CLI, `config.json`, periodic `eval_metrics.csv`, and final full-game JSON. |

## Double DQN

Status: correct Double DQN target implementation, with the same experiment-reporting gap as DQN.

| Checklist item | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| DQN base architecture | Match | Minor | Keeps DQN-style CNN, Atari wrappers, replay, target network, and RMSProp: [double_dqn.py:72](../../../baselines/model-free/DDQN/double_dqn.py#L72), [double_dqn.py:144](../../../baselines/model-free/DDQN/double_dqn.py#L144), [double_dqn.py:173](../../../baselines/model-free/DDQN/double_dqn.py#L173), [double_dqn.py:175](../../../baselines/model-free/DDQN/double_dqn.py#L175). | Keep DQN parity. |
| Online network selects next action | Match | Minor | Uses `q_network(next_states).argmax`: [double_dqn.py:261](../../../baselines/model-free/DDQN/double_dqn.py#L261). | Keep. |
| Target network evaluates selected action | Match | Minor | Uses target network `gather` for the selected action: [double_dqn.py:265](../../../baselines/model-free/DDQN/double_dqn.py#L265), [double_dqn.py:269](../../../baselines/model-free/DDQN/double_dqn.py#L269). | Keep. |
| No unintended drift from DQN | Match | Minor | Architecture and training loop mirror DQN except the target calculation. | Keep parity and state this in docs. |
| Evaluation and reproducibility | Missing | Major | Same single-run reporting limitation as DQN: [double_dqn.py:315](../../../baselines/model-free/DDQN/double_dqn.py#L315), [double_dqn.py:320](../../../baselines/model-free/DDQN/double_dqn.py#L320), [double_dqn.py:384](../../../baselines/model-free/DDQN/double_dqn.py#L384). | Add the same structured evaluation/config outputs as DQN. |

## PER DDQN

Status: correct paper-oriented PER + DDQN implementation, with reporting fixes needed.

| Checklist item | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| DDQN base architecture | Match | Minor | Uses DQN CNN/wrappers and Double DQN target path: [per_ddqn.py:71](../../../baselines/model-free/PER/per_ddqn.py#L71), [per_ddqn.py:225](../../../baselines/model-free/PER/per_ddqn.py#L225), [per_ddqn.py:332](../../../baselines/model-free/PER/per_ddqn.py#L332), [per_ddqn.py:333](../../../baselines/model-free/PER/per_ddqn.py#L333). | Keep. |
| Proportional prioritized replay and SumTree | Match | Minor | SumTree-backed prioritized replay: [per_ddqn.py:90](../../../baselines/model-free/PER/per_ddqn.py#L90), [per_ddqn.py:149](../../../baselines/model-free/PER/per_ddqn.py#L149), [per_ddqn.py:176](../../../baselines/model-free/PER/per_ddqn.py#L176). | Keep. |
| TD-error priority updates | Match | Minor | Computes TD errors and updates priorities: [per_ddqn.py:338](../../../baselines/model-free/PER/per_ddqn.py#L338), [per_ddqn.py:341](../../../baselines/model-free/PER/per_ddqn.py#L341), [per_ddqn.py:215](../../../baselines/model-free/PER/per_ddqn.py#L215). | Keep. |
| Alpha prioritization | Match | Minor | `alpha=0.6`, priority exponent applied: [per_ddqn.py:153](../../../baselines/model-free/PER/per_ddqn.py#L153), [per_ddqn.py:216](../../../baselines/model-free/PER/per_ddqn.py#L216), [per_ddqn.py:256](../../../baselines/model-free/PER/per_ddqn.py#L256). | Keep. |
| Beta IS correction and annealing | Match | Minor | Beta starts at 0.4, anneals to 1.0, and weights use PER IS formula: [per_ddqn.py:197](../../../baselines/model-free/PER/per_ddqn.py#L197), [per_ddqn.py:199](../../../baselines/model-free/PER/per_ddqn.py#L199), [per_ddqn.py:268](../../../baselines/model-free/PER/per_ddqn.py#L268), [per_ddqn.py:270](../../../baselines/model-free/PER/per_ddqn.py#L270). | Keep. |
| IS-weighted loss | Match | Minor | Per-sample Huber loss is weighted before reduction: [per_ddqn.py:345](../../../baselines/model-free/PER/per_ddqn.py#L345), [per_ddqn.py:346](../../../baselines/model-free/PER/per_ddqn.py#L346). | Keep. |
| Evaluation and reproducibility | Missing | Major | Single-run output path only: [per_ddqn.py:385](../../../baselines/model-free/PER/per_ddqn.py#L385), [per_ddqn.py:390](../../../baselines/model-free/PER/per_ddqn.py#L390), [per_ddqn.py:441](../../../baselines/model-free/PER/per_ddqn.py#L441). | Add structured config/eval outputs. |

## Dreamer V2

Status: compact paper-oriented Dreamer V2 baseline. Core world-model architecture is present, but the Atari protocol is not identical to DQN/AMI.

| Checklist item | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| Image encoder/decoder | Match | Minor | Uses shared ConvEncoder/ConvDecoder: [dreamer_v2.py:91](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L91), [dreamer_v2.py:96](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L96), [dreamer_common.py:68](../../../baselines/model-based/dreamer_common.py#L68), [dreamer_common.py:88](../../../baselines/model-based/dreamer_common.py#L88). | Keep. |
| Discrete RSSM prior/posterior | Match | Minor | Uses factored categorical RSSM with observe/imagine paths: [dreamer_v2.py:94](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L94), [dreamer_common.py:199](../../../baselines/model-based/dreamer_common.py#L199), [dreamer_common.py:240](../../../baselines/model-based/dreamer_common.py#L240), [dreamer_common.py:251](../../../baselines/model-based/dreamer_common.py#L251). | Keep. |
| Sequence replay | Match | Minor | Samples contiguous sequences and avoids episode-boundary crossings: [dreamer_v2.py:31](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L31), [dreamer_v2.py:49](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L49). | Keep. |
| Reconstruction, reward, continuation, KL losses | Match | Minor | World loss combines all expected terms: [dreamer_v2.py:290](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L290), [dreamer_v2.py:294](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L294), [dreamer_v2.py:302](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L302), [dreamer_v2.py:307](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L307). | Keep. |
| KL balancing/free nats | Match | Minor | Balanced KL and free-nats loss are present: [dreamer_v2.py:302](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L302), [dreamer_v2.py:306](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L306), [dreamer_common.py:261](../../../baselines/model-based/dreamer_common.py#L261). | Keep. |
| Actor/value learning from imagined rollouts | Match | Minor | Imagination, lambda returns, actor loss, and value loss are implemented: [dreamer_v2.py:114](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L114), [dreamer_v2.py:148](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L148), [dreamer_v2.py:320](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L320), [dreamer_v2.py:324](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L324), [dreamer_v2.py:332](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L332). | Keep. |
| Atari preprocessing/evaluation assumptions | Acceptable simplification | Major | Uses raw `rgb_array` frames resized to 64x64 RGB and manual FIRE, not the DQN/AMI AtariPreprocessing stack: [dreamer_v2.py:152](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L152), [dreamer_v2.py:168](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L168), [dreamer_v2.py:231](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L231), [dreamer_v2.py:376](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L376). | Either document as compact Dreamer protocol or add optional standardized Breakout eval. |
| Structured reporting | Missing | Major | No AMI-style config/eval/final JSON outputs: [dreamer_v2.py:360](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L360), [dreamer_v2.py:375](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L375), [dreamer_v2.py:449](../../../baselines/model-based/Dreamer-v2/dreamer_v2.py#L449). | Add structured outputs and seed CLI. |

## Dreamer V3

Status: architecturally recognizable as compact Dreamer V3-style, but not final-run ready because current logs show an OOM failure.

| Checklist item | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| Dreamer-style world model and imagination actor-critic | Match | Minor | Encoder, discrete RSSM, decoder, reward, continuation, value, actor, and imagined behavior update are present: [dreamer_v3.py:83](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L83), [dreamer_v3.py:89](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L89), [dreamer_v3.py:92](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L92), [dreamer_v3.py:96](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L96), [dreamer_v3.py:333](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L333). | Keep core architecture. |
| Discrete latent dynamics | Match | Minor | Uses `DiscreteRSSM` with 32 stochastic variables and 32 classes: [dreamer_v3.py:89](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L89), [dreamer_common.py:199](../../../baselines/model-based/dreamer_common.py#L199). | Keep. |
| Continuation head | Match | Minor | Continuation head and BCE loss exist: [dreamer_v3.py:94](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L94), [dreamer_v3.py:307](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L307), [dreamer_v3.py:309](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L309). | Keep. |
| Symlog/symexp reward handling | Match | Minor | Reward head trains with symlog targets and imagined rewards use symexp: [dreamer_v3.py:20](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L20), [dreamer_v3.py:136](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L136), [dreamer_v3.py:306](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L306). | Keep. |
| Return normalization | Match | Minor | ReturnNormalizer scales actor advantage: [dreamer_v3.py:150](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L150), [dreamer_v3.py:251](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L251), [dreamer_v3.py:339](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L339). | Keep. |
| Split/balanced KL | Acceptable simplification | Minor | Separate dyn/rep KL losses with free nats are present: [dreamer_v3.py:317](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L317), [dreamer_v3.py:320](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L320), [dreamer_v3.py:322](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L322). | Document as compact approximation. |
| Full Dreamer V3 recipe fidelity | Acceptable simplification | Documentation | Defaults and model are compact/repo-specific: [dreamer_v3.py:431](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L431), [dreamer_v3.py:433](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L433), [dreamer_v3.py:434](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L434). | Use "Dreamer V3-style compact baseline" wording unless expanded. |
| Atari preprocessing/evaluation assumptions | Acceptable simplification | Major | Uses raw `rgb_array` frames resized to 64x64 RGB and manual FIRE: [dreamer_v3.py:177](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L177), [dreamer_v3.py:193](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L193), [dreamer_v3.py:241](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L241), [dreamer_v3.py:392](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L392). | Document protocol or add standardized evaluation. |
| Memory/compute risk | Incorrect | Critical | Current scheduler log reached only partial updates and later failed with CUDA OOM; latest report marks Dreamer V3 failed: [dreamer_v3.log:49](../../../results/scheduler_runs/run_20260428_203450/dreamer_v3.log#L49), [report.md](report.md). | Add safe preset or memory guard before final comparison. |
| Structured reporting | Missing | Major | No AMI-style config/eval/final JSON outputs: [dreamer_v3.py:376](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L376), [dreamer_v3.py:391](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L391), [dreamer_v3.py:461](../../../baselines/model-based/Dreamer-v3/dreamer_v3.py#L461). | Add structured outputs and seed CLI. |

## Out Of Scope Baselines

These are model-based baselines in the repo, but they are not part of the Breakout architecture audit because their current environments differ:

| Baseline | Current environment | Reason |
|---|---|---|
| Dreamer V1 | `dm_control/walker-walk-v0` | Not a Breakout/Atari comparison in this repo. |
| MuZero | `CartPole-v1` | Current implementation is low-dimensional CartPole, not Breakout pixels. |
| Learned Dynamics MPC | `Pendulum-v1` | Continuous-control Pendulum baseline, not Breakout. |
| Dyna-Q | `CliffWalking-v1` | Tabular grid-world baseline, not Breakout. |

## Conclusion

The model-free Breakout baselines are architecturally sound enough for a paper-oriented FYP comparison, but need structured reporting and reproducibility patches. Dreamer V2 is architecturally recognizable and usable as a compact model-based comparator if its preprocessing protocol is clearly documented. Dreamer V3 should not be treated as a completed baseline until the CUDA OOM risk is addressed and a successful run is produced.
