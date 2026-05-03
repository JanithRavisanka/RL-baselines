#!/usr/bin/env python3
"""
Run Breakout baselines used by the LeCun-AMI experiment.

This launcher covers the existing model-free Atari baselines and the model-based
Dreamer baselines that target ALE/Breakout-v5 in this repository.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments" / "lecun_ami_breakout" / "runs"


@dataclass(frozen=True)
class Baseline:
    key: str
    label: str
    family: str
    script: Path
    args: tuple[str, ...] = ()


BASELINES = {
    "dqn": Baseline(
        key="dqn",
        label="DQN",
        family="model-free",
        script=REPO_ROOT / "baselines" / "model-free" / "DQN" / "dqn.py",
    ),
    "ddqn": Baseline(
        key="ddqn",
        label="DDQN",
        family="model-free",
        script=REPO_ROOT / "baselines" / "model-free" / "DDQN" / "double_dqn.py",
    ),
    "per_ddqn": Baseline(
        key="per_ddqn",
        label="PER DDQN",
        family="model-free",
        script=REPO_ROOT / "baselines" / "model-free" / "PER" / "per_ddqn.py",
    ),
    "dreamer_v2": Baseline(
        key="dreamer_v2",
        label="Dreamer V2",
        family="model-based",
        script=REPO_ROOT / "baselines" / "model-based" / "Dreamer-v2" / "dreamer_v2.py",
        args=("--env", "ALE/Breakout-v5"),
    ),
    "dreamer_v3": Baseline(
        key="dreamer_v3",
        label="Dreamer V3",
        family="model-based",
        script=REPO_ROOT / "baselines" / "model-based" / "Dreamer-v3" / "dreamer_v3.py",
        args=("--env", "ALE/Breakout-v5"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Breakout baseline scripts.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=sorted(BASELINES),
        default=sorted(BASELINES),
        help="Baseline keys to run.",
    )
    parser.add_argument(
        "--family",
        choices=["all", "model-free", "model-based"],
        default="all",
        help="Filter baselines by family.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining baselines after one fails.",
    )
    return parser.parse_args()


def selected_baselines(args: argparse.Namespace) -> list[Baseline]:
    selected = [BASELINES[key] for key in args.baselines]
    if args.family == "all":
        return selected
    return [baseline for baseline in selected if baseline.family == args.family]


def manifest_fields() -> list[str]:
    return [
        "started_at",
        "finished_at",
        "baseline",
        "label",
        "family",
        "returncode",
        "command",
    ]


def manifest_path() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"baseline_manifest_{timestamp}.csv"


def write_manifest_header(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writeheader()


def append_manifest(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writerow(row)


def command_for(baseline: Baseline) -> list[str]:
    return [sys.executable, str(baseline.script), *baseline.args]


def run_baseline(baseline: Baseline, manifest: Path | None, dry_run: bool) -> int:
    command = command_for(baseline)
    print()
    print(f"[{baseline.family}] {baseline.label}")
    print(" ".join(command))

    if dry_run:
        return 0

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(command, cwd=REPO_ROOT)
    finished_at = dt.datetime.now().isoformat(timespec="seconds")

    if manifest is not None:
        append_manifest(
            manifest,
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "baseline": baseline.key,
                "label": baseline.label,
                "family": baseline.family,
                "returncode": result.returncode,
                "command": json.dumps(command),
            },
        )

    return result.returncode


def main() -> int:
    args = parse_args()
    baselines = selected_baselines(args)
    if not baselines:
        print("No baselines selected.")
        return 0

    manifest = None if args.dry_run else manifest_path()
    if manifest is not None:
        write_manifest_header(manifest)
        print(f"Writing manifest: {manifest.relative_to(REPO_ROOT)}")

    failures = 0
    for baseline in baselines:
        code = run_baseline(baseline, manifest, args.dry_run)
        if code != 0:
            failures += 1
            print(f"FAILED: {baseline.key} exited with {code}")
            if not args.continue_on_failure:
                return code

    if failures:
        print(f"Completed with {failures} failed baseline(s).")
        return 1

    print("Baseline run set completed.")
    if manifest is not None:
        print(f"Manifest: {manifest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
