#!/usr/bin/env python3
"""
Aggregate LeCun-AMI Breakout experiment outputs into report-ready tables.

AMI runs are read from results/lecun_ami_atari/run_*/. Baseline availability is
checked from the existing model-free and model-based Breakout result folders and
scheduler logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
AMI_RESULTS = RESULTS_ROOT / "lecun_ami_atari"
REPORT_DIR = REPO_ROOT / "experiments" / "lecun_ami_breakout" / "reports"


CONDITION_LABELS = {
    ("adaptive", "adaptive"): "AMI Adaptive",
    ("always", "planner"): "AMI Always Planner",
    ("actor", "actor"): "AMI Actor Only",
}


BASELINES = {
    "dqn": {
        "label": "DQN",
        "family": "model-free",
        "env": "ALE/Breakout-v5",
        "script": "baselines/model-free/DQN/dqn.py",
    },
    "ddqn": {
        "label": "DDQN",
        "family": "model-free",
        "env": "ALE/Breakout-v5",
        "script": "baselines/model-free/DDQN/double_dqn.py",
    },
    "per_ddqn": {
        "label": "PER DDQN",
        "family": "model-free",
        "env": "ALE/Breakout-v5",
        "script": "baselines/model-free/PER/per_ddqn.py",
    },
    "dreamer_v2": {
        "label": "Dreamer V2",
        "family": "model-based",
        "env": "ALE/Breakout-v5",
        "script": "baselines/model-based/Dreamer-v2/dreamer_v2.py",
    },
    "dreamer_v3": {
        "label": "Dreamer V3",
        "family": "model-based",
        "env": "ALE/Breakout-v5",
        "script": "baselines/model-based/Dreamer-v3/dreamer_v3.py",
    },
}


OUT_OF_SCOPE_MODEL_BASED = [
    {
        "algorithm": "Dreamer V1",
        "env": "dm_control/walker-walk-v0",
        "reason": "Not a Breakout/Atari comparison in this repo.",
    },
    {
        "algorithm": "MuZero",
        "env": "CartPole-v1",
        "reason": "Current implementation is low-dimensional CartPole, not Breakout pixels.",
    },
    {
        "algorithm": "Learned Dynamics MPC",
        "env": "Pendulum-v1",
        "reason": "Continuous-control Pendulum baseline, not Breakout.",
    },
    {
        "algorithm": "Dyna-Q",
        "env": "CliffWalking-v1",
        "reason": "Tabular grid-world baseline, not Breakout.",
    },
]


@dataclass
class AmiRun:
    run_dir: Path
    condition: str
    seed: int
    total_steps: int
    completed: bool
    eval_reward_mean: float | None
    eval_reward_std: float | None
    eval_planning_rate_mean: float | None
    elapsed_sec: float | None
    reward_last_100_mean: float | None
    planning_rate_last_100_mean: float | None


@dataclass
class PlotOutput:
    reward_curve: Path | None = None
    planning_curve: Path | None = None
    efficiency_bars: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LeCun-AMI Breakout experiment results.")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs that do not have final_summary.json/model.pth yet.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_last_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def condition_from_config(config: dict[str, Any]) -> str | None:
    key = (config.get("planning_mode"), config.get("eval_mode"))
    return CONDITION_LABELS.get(key)


def is_completed_run(run_dir: Path, total_steps: int) -> bool:
    summary = read_json(run_dir / "final_summary.json")
    summary_steps = to_int(summary.get("total_steps"))
    has_final_checkpoint = (run_dir / "model.pth").exists()
    has_final_eval = (run_dir / "final_eval_metrics.json").exists()
    return has_final_checkpoint and has_final_eval and summary_steps == total_steps


def load_ami_runs(env_name: str, total_steps: int, include_incomplete: bool) -> list[AmiRun]:
    runs: list[AmiRun] = []
    if not AMI_RESULTS.exists():
        return runs

    for run_dir in sorted(AMI_RESULTS.glob("run_*")):
        config_doc = read_json(run_dir / "config.json")
        config = config_doc.get("config", config_doc)
        if not config:
            continue
        if config.get("env") != env_name:
            continue
        if to_int(config.get("total_steps")) != total_steps:
            continue

        condition = condition_from_config(config)
        if condition is None:
            continue

        completed = is_completed_run(run_dir, total_steps)
        if not completed and not include_incomplete:
            continue

        final_eval = read_json(run_dir / "final_eval_metrics.json")
        eval_row = read_last_csv_row(run_dir / "eval_metrics.csv")
        final_summary = read_json(run_dir / "final_summary.json")
        training_row = read_last_csv_row(run_dir / "training_log.csv")
        metrics_row = read_last_csv_row(run_dir / "metrics.csv")

        runs.append(
            AmiRun(
                run_dir=run_dir,
                condition=condition,
                seed=to_int(config.get("seed")),
                total_steps=to_int(config.get("total_steps")),
                completed=completed,
                eval_reward_mean=to_float(final_eval.get("eval_reward_mean"))
                or to_float(eval_row.get("eval_reward_mean")),
                eval_reward_std=to_float(final_eval.get("eval_reward_std"))
                or to_float(eval_row.get("eval_reward_std")),
                eval_planning_rate_mean=to_float(final_eval.get("eval_planning_rate_mean"))
                or to_float(eval_row.get("eval_planning_rate_mean")),
                elapsed_sec=to_float(training_row.get("elapsed_sec"))
                or to_float(metrics_row.get("elapsed_sec")),
                reward_last_100_mean=to_float(final_summary.get("reward_last_100_mean")),
                planning_rate_last_100_mean=to_float(final_summary.get("planning_rate_last_100_mean")),
            )
        )

    return runs


def mean_or_blank(values: list[float | None]) -> str:
    clean = [value for value in values if value is not None]
    if not clean:
        return ""
    return f"{mean(clean):.4f}"


def std_or_blank(values: list[float | None]) -> str:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return ""
    return f"{pstdev(clean):.4f}"


def reward_per_planning(run: AmiRun) -> float | None:
    if run.eval_reward_mean is None:
        return None
    planning = run.eval_planning_rate_mean
    if planning is None or planning <= 0:
        return None
    return run.eval_reward_mean / planning


def reward_per_hour(run: AmiRun) -> float | None:
    if run.eval_reward_mean is None or run.elapsed_sec is None or run.elapsed_sec <= 0:
        return None
    return run.eval_reward_mean / (run.elapsed_sec / 3600.0)


def write_ami_run_table(runs: list[AmiRun], output_dir: Path) -> Path:
    path = output_dir / "ami_runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "seed",
                "total_steps",
                "completed",
                "eval_reward_mean",
                "eval_reward_std",
                "eval_planning_rate_mean",
                "elapsed_sec",
                "reward_per_planning_rate",
                "reward_per_wall_clock_hour",
                "reward_last_100_mean",
                "planning_rate_last_100_mean",
                "run_dir",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "condition": run.condition,
                    "seed": run.seed,
                    "total_steps": run.total_steps,
                    "completed": int(run.completed),
                    "eval_reward_mean": "" if run.eval_reward_mean is None else run.eval_reward_mean,
                    "eval_reward_std": "" if run.eval_reward_std is None else run.eval_reward_std,
                    "eval_planning_rate_mean": ""
                    if run.eval_planning_rate_mean is None
                    else run.eval_planning_rate_mean,
                    "elapsed_sec": "" if run.elapsed_sec is None else run.elapsed_sec,
                    "reward_per_planning_rate": ""
                    if reward_per_planning(run) is None
                    else reward_per_planning(run),
                    "reward_per_wall_clock_hour": ""
                    if reward_per_hour(run) is None
                    else reward_per_hour(run),
                    "reward_last_100_mean": ""
                    if run.reward_last_100_mean is None
                    else run.reward_last_100_mean,
                    "planning_rate_last_100_mean": ""
                    if run.planning_rate_last_100_mean is None
                    else run.planning_rate_last_100_mean,
                    "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
                }
            )
    return path


def write_ami_summary(runs: list[AmiRun], output_dir: Path) -> Path:
    grouped: dict[str, list[AmiRun]] = defaultdict(list)
    for run in runs:
        grouped[run.condition].append(run)

    path = output_dir / "ami_condition_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "completed_runs",
                "seeds_completed",
                "reward_mean",
                "reward_std_across_seeds",
                "planning_rate_mean",
                "elapsed_hours_mean",
                "reward_per_planning_rate_mean",
                "reward_per_wall_clock_hour_mean",
            ],
        )
        writer.writeheader()
        for condition in sorted(grouped):
            condition_runs = grouped[condition]
            writer.writerow(
                {
                    "condition": condition,
                    "completed_runs": sum(1 for run in condition_runs if run.completed),
                    "seeds_completed": len({run.seed for run in condition_runs}),
                    "reward_mean": mean_or_blank([run.eval_reward_mean for run in condition_runs]),
                    "reward_std_across_seeds": std_or_blank(
                        [run.eval_reward_mean for run in condition_runs]
                    ),
                    "planning_rate_mean": mean_or_blank(
                        [run.eval_planning_rate_mean for run in condition_runs]
                    ),
                    "elapsed_hours_mean": mean_or_blank(
                        [
                            None if run.elapsed_sec is None else run.elapsed_sec / 3600.0
                            for run in condition_runs
                        ]
                    ),
                    "reward_per_planning_rate_mean": mean_or_blank(
                        [reward_per_planning(run) for run in condition_runs]
                    ),
                    "reward_per_wall_clock_hour_mean": mean_or_blank(
                        [reward_per_hour(run) for run in condition_runs]
                    ),
                }
            )
    return path


def parse_baseline_log(algo: str) -> dict[str, Any]:
    frame_pattern = re.compile(r"Frame:\s+(\d+)/(\d+).*Avg Reward \(Last 20\):\s+([-0-9.]+)")
    update_pattern = re.compile(r"Update\s+(\d+)/(\d+)")
    collect_pattern = re.compile(
        r"Collected\s+(\d+)\s+policy steps.*reward sum\s+([-0-9.]+).*replay\s+(\d+)"
    )
    eval_pattern = re.compile(r"Evaluation finished\. Total Reward:\s+([-0-9.]+)")
    dreamer_eval_pattern = re.compile(r"Evaluation reward:\s+([-0-9.]+)")
    oom_pattern = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
    traceback_pattern = re.compile(r"Traceback")
    best: dict[str, Any] = {}

    for log_path in sorted((RESULTS_ROOT / "scheduler_runs").glob(f"run_*/{algo}.log")):
        last_frame = None
        total_frames = None
        last_update = None
        total_updates = None
        replay_size = None
        final_avg = None
        final_collect_reward = None
        final_eval = None
        failed = False
        with log_path.open(errors="replace") as handle:
            for line in handle:
                frame_match = frame_pattern.search(line)
                if frame_match:
                    last_frame = int(frame_match.group(1))
                    total_frames = int(frame_match.group(2))
                    final_avg = float(frame_match.group(3))
                update_match = update_pattern.search(line)
                if update_match:
                    last_update = int(update_match.group(1))
                    total_updates = int(update_match.group(2))
                collect_match = collect_pattern.search(line)
                if collect_match:
                    final_collect_reward = float(collect_match.group(2))
                    replay_size = int(collect_match.group(3))
                eval_match = eval_pattern.search(line)
                if eval_match:
                    final_eval = float(eval_match.group(1))
                dreamer_eval_match = dreamer_eval_pattern.search(line)
                if dreamer_eval_match:
                    final_eval = float(dreamer_eval_match.group(1))
                if traceback_pattern.search(line) or oom_pattern.search(line):
                    failed = True

        progress = last_frame if last_frame is not None else last_update or 0
        if progress and (not best or progress >= best.get("progress", 0)):
            best = {
                "log_path": log_path,
                "progress": progress,
                "last_frame": last_frame,
                "total_frames": total_frames,
                "last_update": last_update,
                "total_updates": total_updates,
                "replay_size": replay_size,
                "final_avg_reward_last_20": final_avg,
                "final_collect_reward_sum": final_collect_reward,
                "final_eval_reward": final_eval,
                "failed": failed,
            }
    return best


def baseline_rows() -> list[dict[str, Any]]:
    rows = []
    for algo, info in BASELINES.items():
        algo_dir = RESULTS_ROOT / algo
        run_dirs = sorted(path for path in algo_dir.glob("run_*") if path.is_dir()) if algo_dir.exists() else []
        latest_run = run_dirs[-1] if run_dirs else None
        log = parse_baseline_log(algo)
        rows.append(
            {
                "algorithm": info["label"],
                "family": info["family"],
                "env": info["env"],
                "script": info["script"],
                "latest_result_dir": ""
                if latest_run is None
                else str(latest_run.relative_to(REPO_ROOT)),
                "has_training_curve": bool(latest_run and (latest_run / "training_curve.png").exists()),
                "has_gif": bool(latest_run and list(latest_run.glob("*.gif"))),
                "log_path": "" if not log else str(log["log_path"].relative_to(REPO_ROOT)),
                "last_logged_frame": "" if not log else log.get("last_frame", ""),
                "total_frames": "" if not log else log.get("total_frames", ""),
                "last_logged_update": "" if not log else log.get("last_update", ""),
                "total_updates": "" if not log else log.get("total_updates", ""),
                "replay_size": "" if not log else log.get("replay_size", ""),
                "final_avg_reward_last_20": "" if not log else log.get("final_avg_reward_last_20", ""),
                "final_collect_reward_sum": "" if not log else log.get("final_collect_reward_sum", ""),
                "final_eval_reward": "" if not log else log.get("final_eval_reward", ""),
                "failed": "" if not log else int(bool(log.get("failed"))),
            }
        )
    return rows


def write_baseline_summary(output_dir: Path) -> Path:
    path = output_dir / "baseline_availability.csv"
    rows = baseline_rows()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_out_of_scope_model_based(output_dir: Path) -> Path:
    path = output_dir / "model_based_out_of_scope.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUT_OF_SCOPE_MODEL_BASED[0].keys()))
        writer.writeheader()
        writer.writerows(OUT_OF_SCOPE_MODEL_BASED)
    return path


def eval_series(run: AmiRun) -> list[dict[str, float]]:
    path = run.run_dir / "eval_metrics.csv"
    if not path.exists():
        return []
    series = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = to_float(row.get("global_step"))
            reward = to_float(row.get("eval_reward_mean"))
            planning = to_float(row.get("eval_planning_rate_mean"))
            if step is None:
                continue
            series.append(
                {
                    "global_step": step,
                    "eval_reward_mean": reward if reward is not None else float("nan"),
                    "eval_planning_rate_mean": planning if planning is not None else float("nan"),
                }
            )
    return series


def write_plots(runs: list[AmiRun], output_dir: Path) -> PlotOutput:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping report plots because matplotlib could not be imported: {exc}")
        return PlotOutput()

    outputs = PlotOutput()
    plotted_reward = False
    plotted_planning = False

    plt.figure(figsize=(10, 5))
    for run in runs:
        series = eval_series(run)
        if not series:
            continue
        steps = [row["global_step"] for row in series]
        rewards = [row["eval_reward_mean"] for row in series]
        if all(math.isnan(value) for value in rewards):
            continue
        plt.plot(steps, rewards, marker="o", label=f"{run.condition} seed {run.seed}")
        plotted_reward = True
    if plotted_reward:
        plt.title("Full-game evaluation reward")
        plt.xlabel("Agent step")
        plt.ylabel("Mean reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outputs.reward_curve = output_dir / "ami_eval_reward_curve.png"
        plt.savefig(outputs.reward_curve, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for run in runs:
        series = eval_series(run)
        if not series:
            continue
        steps = [row["global_step"] for row in series]
        planning = [row["eval_planning_rate_mean"] for row in series]
        if all(math.isnan(value) for value in planning):
            continue
        plt.plot(steps, planning, marker="o", label=f"{run.condition} seed {run.seed}")
        plotted_planning = True
    if plotted_planning:
        plt.title("Full-game evaluation planning rate")
        plt.xlabel("Agent step")
        plt.ylabel("Fraction of planned steps")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        outputs.planning_curve = output_dir / "ami_eval_planning_rate_curve.png"
        plt.savefig(outputs.planning_curve, bbox_inches="tight")
    plt.close()

    grouped: dict[str, list[AmiRun]] = defaultdict(list)
    for run in runs:
        grouped[run.condition].append(run)
    labels = []
    reward_per_hour_values = []
    reward_per_planning_values = []
    for condition in sorted(grouped):
        per_hour = [reward_per_hour(run) for run in grouped[condition]]
        per_planning = [reward_per_planning(run) for run in grouped[condition]]
        per_hour_clean = [value for value in per_hour if value is not None]
        per_planning_clean = [value for value in per_planning if value is not None]
        if per_hour_clean or per_planning_clean:
            labels.append(condition)
            reward_per_hour_values.append(mean(per_hour_clean) if per_hour_clean else 0.0)
            reward_per_planning_values.append(mean(per_planning_clean) if per_planning_clean else 0.0)

    if labels:
        x = list(range(len(labels)))
        width = 0.38
        plt.figure(figsize=(10, 5))
        plt.bar([value - width / 2 for value in x], reward_per_hour_values, width, label="Reward/hour")
        plt.bar(
            [value + width / 2 for value in x],
            reward_per_planning_values,
            width,
            label="Reward/planning rate",
        )
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title("AMI reward efficiency")
        plt.ylabel("Efficiency score")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        outputs.efficiency_bars = output_dir / "ami_efficiency_bars.png"
        plt.savefig(outputs.efficiency_bars, bbox_inches="tight")
        plt.close()

    return outputs


def write_markdown_report(runs: list[AmiRun], output_dir: Path, plots: PlotOutput) -> Path:
    summary_path = output_dir / "ami_condition_summary.csv"
    baseline_path = output_dir / "baseline_availability.csv"
    out_of_scope_path = output_dir / "model_based_out_of_scope.csv"
    markdown_path = output_dir / "report.md"

    with summary_path.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with baseline_path.open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    with out_of_scope_path.open(newline="") as handle:
        out_of_scope = list(csv.DictReader(handle))

    lines = [
        "# LeCun-AMI Breakout Experiment Report",
        "",
        "## AMI Conditions",
        "",
        "| Condition | Seeds | Reward mean | Reward std | Planning rate | Hours mean | Reward/planning | Reward/hour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {seeds_completed} | {reward_mean} | {reward_std_across_seeds} | "
            "{planning_rate_mean} | {elapsed_hours_mean} | {reward_per_planning_rate_mean} | "
            "{reward_per_wall_clock_hour_mean} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Breakout Baseline Availability",
            "",
            "| Family | Algorithm | Result dir | Frame/update progress | Last-20 reward | Collect reward sum | Eval reward | Failed |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in baseline:
        progress = row["last_logged_frame"] or row["last_logged_update"]
        progress_total = row["total_frames"] or row["total_updates"]
        progress_text = progress if not progress_total else f"{progress}/{progress_total}"
        lines.append(
            "| {family} | {algorithm} | {latest_result_dir} | {progress} | "
            "{final_avg_reward_last_20} | {final_collect_reward_sum} | "
            "{final_eval_reward} | {failed} |".format(**row, progress=progress_text)
        )

    lines.extend(
        [
            "",
            "## Model-Based Baselines Not Used For Breakout",
            "",
            "| Algorithm | Environment | Reason |",
            "|---|---|---|",
        ]
    )
    for row in out_of_scope:
        lines.append(
            "| {algorithm} | {env} | {reason} |".format(**row)
        )

    plot_lines = []
    for title, path in [
        ("Evaluation reward curve", plots.reward_curve),
        ("Planning-rate curve", plots.planning_curve),
        ("Efficiency bars", plots.efficiency_bars),
    ]:
        if path is not None:
            plot_lines.append(f"- [{title}]({path.name})")
    if plot_lines:
        lines.extend(["", "## Plots", "", *plot_lines])

    lines.extend(
        [
            "",
            "## Conclusion Template",
            "",
            "Adaptive planning improved: TODO after all seeds complete.",
            "",
            "Compute efficiency improved: TODO after comparing reward/hour and planning rate.",
            "",
            "Notes: AMI uses structured full-game evaluation logs. Existing DQN/DDQN/PER and Dreamer V2/V3 baselines are summarized from artifacts and scheduler logs, so compare them cautiously unless their scripts are upgraded to emit the same periodic evaluation CSV.",
            "",
            f"AMI runs included: {len(runs)}",
        ]
    )

    markdown_path.write_text("\n".join(lines) + "\n")
    return markdown_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_ami_runs(args.env, args.total_steps, args.include_incomplete)
    run_table = write_ami_run_table(runs, args.output_dir)
    summary = write_ami_summary(runs, args.output_dir)
    baseline = write_baseline_summary(args.output_dir)
    out_of_scope = write_out_of_scope_model_based(args.output_dir)
    plots = write_plots(runs, args.output_dir)
    report = write_markdown_report(runs, args.output_dir, plots)

    print(f"AMI runs summarized: {len(runs)}")
    print(f"Wrote {run_table.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary.relative_to(REPO_ROOT)}")
    print(f"Wrote {baseline.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_of_scope.relative_to(REPO_ROOT)}")
    print(f"Wrote {report.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
