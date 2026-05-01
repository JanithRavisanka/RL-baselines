#!/usr/bin/env python3
"""
Optimized staged launcher for the RTX PRO 4000 / 24-core server.

This script does not change any learning algorithm. It only:
- caps BLAS/OpenMP thread fan-out per Python process,
- runs compatible algorithms in stages,
- avoids running CPU-heavy env/MCTS jobs beside GPU-heavy Dreamer jobs.

By default, `run_all_algorithms.py` still skips completed runs. Use `--force`
only when you intentionally want to retrain everything from scratch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Stage:
    name: str
    algos: tuple[str, ...]
    cpu_slots: int
    gpu_budget_mb: int


STAGES = (
    Stage(
        name="light",
        algos=("dyna_q", "ppo", "ddpg", "td3", "sac", "mpc"),
        cpu_slots=10,
        gpu_budget_mb=3000,
    ),
    Stage(
        name="atari_value",
        algos=("dqn", "ddqn", "per_ddqn"),
        cpu_slots=8,
        gpu_budget_mb=3000,
    ),
    Stage(
        name="muzero",
        algos=("muzero",),
        cpu_slots=12,
        gpu_budget_mb=3000,
    ),
    Stage(
        name="dreamer_v1",
        algos=("dreamer_v1",),
        cpu_slots=6,
        gpu_budget_mb=24000,
    ),
    Stage(
        name="dreamer_v2",
        algos=("dreamer_v2",),
        cpu_slots=6,
        gpu_budget_mb=24000,
    ),
    Stage(
        name="dreamer_v3",
        algos=("dreamer_v3",),
        cpu_slots=6,
        gpu_budget_mb=24000,
    ),
)


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def stage_by_name(names: Iterable[str]) -> list[Stage]:
    requested = list(names)
    available = {stage.name: stage for stage in STAGES}
    unknown = [name for name in requested if name not in available]
    if unknown:
        valid = ", ".join(available)
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Valid stages: {valid}")
    return [available[name] for name in requested]


def build_command(repo_root: Path, stage: Stage, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(repo_root / "run_all_algorithms.py"),
        "--only",
        *stage.algos,
        "--cpu-slots",
        str(stage.cpu_slots),
        "--gpu-budget-mb",
        str(stage.gpu_budget_mb),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.force:
        cmd.append("--no-skip")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def print_plan(stages: list[Stage], force: bool, dry_run: bool):
    mode = "dry-run" if dry_run else "execute"
    skip_mode = "force rerun" if force else "resume/skip completed"
    print(f"Optimized server launcher: {mode}, {skip_mode}")
    print("Thread caps: " + ", ".join(f"{k}={v}" for k, v in THREAD_ENV.items() if k != "PYTHONUNBUFFERED"))
    print()
    for idx, stage in enumerate(stages, start=1):
        algos = " ".join(stage.algos)
        print(
            f"{idx}. {stage.name:<12s} "
            f"cpu_slots={stage.cpu_slots:<2d} gpu_budget_mb={stage.gpu_budget_mb:<5d} algos={algos}"
        )
    print()


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run RL baselines in server-optimized stages without changing algorithm code."
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=[stage.name for stage in STAGES],
        help="Run only selected stage(s). Default: all stages in optimized order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --no-skip to run_all_algorithms.py and retrain even if outputs exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each stage's run_all_algorithms.py dry-run plan without launching training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stages if one stage fails.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
        help="Polling interval passed to run_all_algorithms.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    args = parser.parse_args()

    stages = stage_by_name(args.stage) if args.stage else list(STAGES)
    print_plan(stages, force=args.force, dry_run=args.dry_run)
    sys.stdout.flush()

    env = os.environ.copy()
    env.update(THREAD_ENV)

    failures = []
    for stage in stages:
        cmd = build_command(repo_root, stage, args)
        print("=" * 80)
        print(f"Starting stage: {stage.name}")
        print("Command: " + " ".join(cmd))
        print("=" * 80)
        sys.stdout.flush()
        result = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
        if result.returncode != 0:
            failures.append((stage.name, result.returncode))
            print(f"Stage failed: {stage.name} exit={result.returncode}")
            if not args.continue_on_error:
                break

    if failures:
        print()
        print("Failed stages:")
        for stage_name, code in failures:
            print(f"- {stage_name}: exit={code}")
        return 1

    print()
    print("All requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
