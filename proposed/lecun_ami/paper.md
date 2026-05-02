# Method Notes

Primary inspiration:

- Yann LeCun, "A Path Towards Autonomous Machine Intelligence", 2022.

The proposal in that paper argues for agents built from modules such as:

- a world model,
- a cost module,
- an actor,
- short-term memory,
- perception,
- and a configurator.

This folder implements a small RL version of that idea:

```text
observation -> encoder -> latent state
latent state + action -> predicted future latent state
predicted future latent state -> cost/value estimate
configurator -> actor or planner
```

The implementation uses a JEPA-style latent prediction objective rather than
pixel reconstruction. This keeps the model compact and directly useful for
planning.

