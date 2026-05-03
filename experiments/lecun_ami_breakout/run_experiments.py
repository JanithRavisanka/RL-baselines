#!/usr/bin/env python3
"""
Run the FYP LeCun-AMI Breakout experiment matrix.

The script intentionally keeps long-running training explicit:

  - smoke: validates the three AMI planning conditions quickly.
  - main: runs the 500k-step, 3-seed AMI matrix from the experiment plan.

It writes a manifest under experiments/lecun_ami_breakout/runs/ so every run can
be traced back to the exact command and output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
AMI_SCRIPT = REPO_ROOT / "proposed" / "lecun_ami" / "lecun_ami_atari.py"
AMI_RESULTS = REPO_ROOT / "results" / "lecun_ami_atari"
RUNS_DIR = REPO_ROOT / "experiments" / "lecun_ami_breakout" / "runs"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    planning_mode: str
    eval_mode: str


CONDITIONS = {
    "adaptive": Condition(
        key="adaptive",
        label="AMI Adaptive",
        planning_mode="adaptive",
        eval_mode="adaptive",
    ),
    "always": Condition(
        key="always",
        label="AMI Always Planner",
        planning_mode="always",
        eval_mode="planner",
    ),
    "actor": Condition(
        key="actor",
        label="AMI Actor Only",
        planning_mode="actor",
        eval_mode="actor",
    ),
}


MAIN_DEFAULTS = {
    "env": "ALE/Breakout-v5",
    "total_steps": 500_000,
    "seed_steps": 10_000,
    "initial_updates": 2_000,
    "batch_size": 32,
    "planning_interval": 4,
    "uncertainty_threshold": 0.001,
    "num_sequences": 512,
    "horizon": 12,
    "critic_n_step": 5,
    "eval_interval": 50_000,
    "eval_episodes": 5,
    "checkpoint_interval": 100_000,
}


SMOKE_DEFAULTS = {
    **MAIN_DEFAULTS,
    "total_steps": 1_000,
    "seed_steps": 200,
    "initial_updates": 20,
    "batch_size": 16,
    "num_sequences": 32,
    "horizon": 4,
    "eval_interval": 500,
    "eval_episodes": 1,
    "checkpoint_interval": 0,
    "no_plots": True,
    "no_gif": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LeCun-AMI Breakout smoke or main experiment matrix."
    )
    parser.add_argument(
        "--phase",
        choices=["smoke", "main", "both"],
        default="smoke",
        help="Experiment phase to run. Use smoke before starting the long main jobs.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=sorted(CONDITIONS),
        help="AMI planning conditions to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seeds for main runs. Smoke uses --smoke-seed unless --phase main.",
    )
    parser.add_argument(
        "--smoke-seed",
        type=int,
        default=1,
        help="Single seed used for smoke validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write no manifest rows.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining jobs after a failure instead of stopping immediately.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra CLI argument passed to lecun_ami_atari.py. Repeat for multiple args.",
    )
    return parser.parse_args()


def snake_to_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_command(condition: Condition, seed: int, defaults: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        str(AMI_SCRIPT),
        "--planning-mode",
        condition.planning_mode,
        "--eval-mode",
        condition.eval_mode,
        "--seed",
        str(seed),
    ]

    for key, value in defaults.items():
        if isinstance(value, bool):
            if value:
                command.append(snake_to_flag(key))
            continue
        command.extend([snake_to_flag(key), str(value)])

    return command


def phases_to_run(phase: str) -> list[str]:
    if phase == "both":
        return ["smoke", "main"]
    return [phase]


def jobs(args: argparse.Namespace) -> Iterable[tuple[str, Condition, int, dict[str, object]]]:
    for phase in phases_to_run(args.phase):
        defaults = SMOKE_DEFAULTS if phase == "smoke" else MAIN_DEFAULTS
        seeds = [args.smoke_seed] if phase == "smoke" else args.seeds
        for condition_key in args.conditions:
            for seed in seeds:
                yield phase, CONDITIONS[condition_key], seed, defaults


def latest_run_dirs() -> set[Path]:
    if not AMI_RESULTS.exists():
        return set()
    return {path for path in AMI_RESULTS.glob("run_*") if path.is_dir()}


def detect_new_run(before: set[Path]) -> Path | None:
    after = latest_run_dirs()
    created = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    return created[0] if created else None


def manifest_path() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"manifest_{timestamp}.csv"


def write_manifest_header(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writeheader()


def manifest_fields() -> list[str]:
    return [
        "started_at",
        "finished_at",
        "phase",
        "condition",
        "label",
        "seed",
        "planning_mode",
        "eval_mode",
        "returncode",
        "run_dir",
        "command",
    ]


def append_manifest(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writerow(row)


def print_command(command: list[str]) -> None:
    print(" ".join(command))


def run_job(
    phase: str,
    condition: Condition,
    seed: int,
    defaults: dict[str, object],
    args: argparse.Namespace,
    manifest: Path | None,
) -> int:
    command = build_command(condition, seed, defaults) + args.extra_arg
    print()
    print(f"[{phase}] {condition.label} | seed={seed}")
    print_command(command)

    if args.dry_run:
        return 0

    before = latest_run_dirs()
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(command, cwd=REPO_ROOT)
    finished_at = dt.datetime.now().isoformat(timespec="seconds")
    run_dir = detect_new_run(before)

    if manifest is not None:
        append_manifest(
            manifest,
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "phase": phase,
                "condition": condition.key,
                "label": condition.label,
                "seed": seed,
                "planning_mode": condition.planning_mode,
                "eval_mode": condition.eval_mode,
                "returncode": result.returncode,
                "run_dir": "" if run_dir is None else str(run_dir.relative_to(REPO_ROOT)),
                "command": json.dumps(command),
            },
        )

    return result.returncode


def main() -> int:
    args = parse_args()
    manifest = None if args.dry_run else manifest_path()
    if manifest is not None:
        write_manifest_header(manifest)
        print(f"Writing manifest: {manifest.relative_to(REPO_ROOT)}")

    failures = 0
    for phase, condition, seed, defaults in jobs(args):
        code = run_job(phase, condition, seed, defaults, args, manifest)
        if code != 0:
            failures += 1
            print(f"FAILED: {phase} {condition.key} seed={seed} exited with {code}")
            if not args.continue_on_failure:
                return code

    if failures:
        print(f"Completed with {failures} failed job(s).")
        return 1

    print("Experiment matrix completed.")
    if manifest is not None:
        print(f"Manifest: {manifest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
