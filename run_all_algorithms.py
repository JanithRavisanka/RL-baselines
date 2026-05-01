#!/usr/bin/env python3
"""
Resource-aware parallel scheduler for RL baseline algorithms.

Instead of a flat --max-parallel cap, this scheduler tracks GPU VRAM budget
and CPU slot budget so that:
  - CPU-only jobs run freely alongside GPU jobs (no wasted GPU idle time).
  - Multiple light GPU jobs can co-run when VRAM permits.
  - Heavy GPU jobs get near-exclusive GPU access automatically.
  - Shortest-job-first ordering frees resources sooner.
  - Already-completed algorithms are skipped on re-runs.

All algorithm hyperparameters, training loops, and model architectures are
completely untouched — only the orchestration layer is smarter.
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Job definition with resource annotations
# ---------------------------------------------------------------------------

@dataclass
class Job:
    name: str
    script_path: Path
    # Resource profile
    gpu_vram_mb: int = 0        # 0 = CPU-only
    cpu_weight: int = 1         # 1 = light, 2 = medium, 4 = heavy
    estimated_minutes: int = 5  # rough expected runtime
    extra_args: Optional[List[str]] = None
    # Result detection patterns (glob relative to results/<name>/)
    result_patterns: List[str] = field(default_factory=lambda: ["model.pth"])


def build_jobs(repo_root: Path) -> List[Job]:
    return [
        # --- Model-Free ---
        Job(
            "actor_critic",
            repo_root / "baselines/model-free/actor-critic/actor_critic.py",
            gpu_vram_mb=0,       # CPU-only, tiny MLP on CartPole
            cpu_weight=1,
            estimated_minutes=2,
        ),
        Job(
            "actor_critic_continuous",
            repo_root / "baselines/model-free/actor-critic/actor_critic_continuous.py",
            gpu_vram_mb=100,     # small MLP on GPU, Pendulum
            cpu_weight=1,
            estimated_minutes=30,
        ),
        Job(
            "dqn",
            repo_root / "baselines/model-free/DQN/dqn.py",
            gpu_vram_mb=250,     # CNN on Atari, 500k frames
            cpu_weight=2,
            estimated_minutes=30,
        ),
        Job(
            "ddqn",
            repo_root / "baselines/model-free/DDQN/double_dqn.py",
            gpu_vram_mb=250,     # CNN on Atari, 500k frames
            cpu_weight=2,
            estimated_minutes=30,
        ),
        Job(
            "per_ddqn",
            repo_root / "baselines/model-free/PER/per_ddqn.py",
            gpu_vram_mb=250,     # CNN on Atari + SumTree, 500k frames
            cpu_weight=2,
            estimated_minutes=30,
        ),
        # --- Model-Based ---
        Job(
            "dyna_q",
            repo_root / "baselines/model-based/Dyna-Q/dyna_q.py",
            gpu_vram_mb=0,       # Tabular, pure CPU
            cpu_weight=1,
            estimated_minutes=5,
            result_patterns=["q_table.npy"],
        ),
        Job(
            "mpc",
            repo_root / "baselines/model-based/MPC/learned_dynamics_mpc.py",
            gpu_vram_mb=150,     # small ensemble of MLPs
            cpu_weight=2,
            estimated_minutes=20,
            result_patterns=["dynamics_ensemble.pth"],
        ),
        Job(
            "muzero",
            repo_root / "baselines/model-based/MuZero/muzero.py",
            gpu_vram_mb=250,     # MLP + MCTS (CPU-heavy)
            cpu_weight=4,
            estimated_minutes=240,
            result_patterns=["muzero_network.pth"],
        ),
        Job(
            "dreamer_v1",
            repo_root / "baselines/model-based/Dreamer-v1/dreamer_v1.py",
            gpu_vram_mb=6500,    # ConvEncoder/Decoder + RSSM, pixel-based
            cpu_weight=2,
            estimated_minutes=120,
        ),
        Job(
            "dreamer_v2",
            repo_root / "baselines/model-based/Dreamer-v2/dreamer_v2.py",
            gpu_vram_mb=5000,    # larger RSSM (deter=600, stoch=32x32)
            cpu_weight=2,
            estimated_minutes=240,
        ),
        Job(
            "dreamer_v3",
            repo_root / "baselines/model-based/Dreamer-v3/dreamer_v3.py",
            gpu_vram_mb=4500,    # largest model (deter=1024, stoch=32x32)
            cpu_weight=3,
            estimated_minutes=360,
        ),
    ]


# ---------------------------------------------------------------------------
# Resource budget tracker
# ---------------------------------------------------------------------------

class ResourceBudget:
    """Tracks available GPU VRAM and CPU slots."""

    def __init__(self, gpu_budget_mb: int, cpu_slots: int):
        self.gpu_total = gpu_budget_mb
        self.cpu_total = cpu_slots
        self.gpu_used = 0
        self.cpu_used = 0

    def can_fit(self, job: Job) -> bool:
        gpu_ok = (self.gpu_used + job.gpu_vram_mb) <= self.gpu_total
        cpu_ok = (self.cpu_used + job.cpu_weight) <= self.cpu_total
        return gpu_ok and cpu_ok

    def allocate(self, job: Job):
        self.gpu_used += job.gpu_vram_mb
        self.cpu_used += job.cpu_weight

    def release(self, job: Job):
        self.gpu_used -= job.gpu_vram_mb
        self.cpu_used -= job.cpu_weight

    def status(self) -> str:
        return (
            f"GPU: {self.gpu_used}/{self.gpu_total} MB  |  "
            f"CPU slots: {self.cpu_used}/{self.cpu_total}"
        )


# ---------------------------------------------------------------------------
# Skip-completed detection
# ---------------------------------------------------------------------------

def is_already_completed(job: Job, results_root: Path) -> bool:
    """Check if any run directory for this algorithm already has results."""
    algo_dir = results_root / job.name
    if not algo_dir.exists():
        return False
    for run_dir in sorted(algo_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for pattern in job.result_patterns:
            if list(run_dir.glob(pattern)):
                return True
    return False


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds + td.days * 86400, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_dashboard(running, queued, completed, skipped, budget, start_time):
    """Print a live status dashboard."""
    elapsed = time.time() - start_time
    print()
    print("=" * 72)
    print(f"  SCHEDULER STATUS  |  Elapsed: {format_duration(elapsed)}  |  {budget.status()}")
    print("=" * 72)

    if running:
        print(f"\n  🟢 RUNNING ({len(running)}):")
        for job, proc, _, _, job_start in running:
            job_elapsed = time.time() - job_start
            gpu_tag = f"GPU:{job.gpu_vram_mb}MB" if job.gpu_vram_mb > 0 else "CPU-only"
            print(f"     {job.name:<30s} {gpu_tag:<14s} {format_duration(job_elapsed)}")

    if queued:
        print(f"\n  🟡 QUEUED ({len(queued)}):")
        for job in queued:
            gpu_tag = f"GPU:{job.gpu_vram_mb}MB" if job.gpu_vram_mb > 0 else "CPU-only"
            print(f"     {job.name:<30s} {gpu_tag:<14s} ~{job.estimated_minutes} min")

    if completed:
        print(f"\n  ✅ COMPLETED ({len(completed)}):")
        for name, code, status, duration in completed:
            icon = "✓" if status == "ok" else "✗"
            dur_str = format_duration(duration) if duration else "—"
            print(f"     {icon} {name:<30s} exit={code:<4s} {dur_str}")

    if skipped:
        print(f"\n  ⏭️  SKIPPED ({len(skipped)}):")
        for name in skipped:
            print(f"     {name}")

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resource-aware parallel scheduler for RL baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_all_algorithms.py                        # smart defaults
  python run_all_algorithms.py --gpu-budget-mb 12000  # reduce GPU budget
  python run_all_algorithms.py --no-skip              # force re-run all
  python run_all_algorithms.py --dry-run              # show plan only
  python run_all_algorithms.py --only dqn muzero      # run specific algos
""",
    )
    parser.add_argument(
        "--gpu-budget-mb", type=int, default=14000,
        help="Total GPU VRAM budget in MB (default: 14000, i.e. ~14GB of 16GB A4000)."
    )
    parser.add_argument(
        "--cpu-slots", type=int, default=10,
        help="Max CPU weight slots to use concurrently (default: 10 of 24 cores)."
    )
    parser.add_argument(
        "--poll-seconds", type=int, default=10,
        help="Polling interval in seconds (default: 10)."
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Force re-run all algorithms even if results already exist."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the scheduling plan without actually launching anything."
    )
    parser.add_argument(
        "--only", nargs="+", metavar="ALGO",
        help="Run only the specified algorithm(s) by name."
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="ALGO",
        help="Exclude specific algorithm(s) by name."
    )
    # Backward-compatible fallback
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="(Legacy) Ignore resource budgets and use a flat parallelism cap."
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    results_root = repo_root / "results"
    all_jobs = build_jobs(repo_root)

    # Filter by --only / --exclude
    if args.only:
        only_set = set(args.only)
        all_jobs = [j for j in all_jobs if j.name in only_set]
        unknown = only_set - {j.name for j in all_jobs}
        if unknown:
            print(f"Warning: unknown algorithm names: {unknown}")
    if args.exclude:
        exclude_set = set(args.exclude)
        all_jobs = [j for j in all_jobs if j.name not in exclude_set]

    # Check for missing scripts
    valid_jobs = []
    for job in all_jobs:
        if not job.script_path.exists():
            print(f"[SKIP] {job.name}: script not found: {job.script_path}")
        else:
            valid_jobs.append(job)

    # Skip already-completed
    skipped_names = []
    if not args.no_skip:
        pending = []
        for job in valid_jobs:
            if is_already_completed(job, results_root):
                skipped_names.append(job.name)
            else:
                pending.append(job)
        valid_jobs = pending

    # Sort by estimated duration (shortest first)
    valid_jobs.sort(key=lambda j: j.estimated_minutes)

    # --- Dry-run mode ---
    if args.dry_run:
        print("\n" + "=" * 72)
        print("  DRY RUN — Scheduling Plan")
        print("=" * 72)
        if skipped_names:
            print(f"\n  Skipped (already completed): {', '.join(skipped_names)}")
        print(f"\n  Jobs to run ({len(valid_jobs)}):\n")
        print(f"  {'#':<4s} {'Name':<30s} {'GPU MB':<10s} {'CPU wt':<8s} {'Est. min':<10s}")
        print(f"  {'-'*4} {'-'*30} {'-'*10} {'-'*8} {'-'*10}")
        for i, job in enumerate(valid_jobs, 1):
            gpu = str(job.gpu_vram_mb) if job.gpu_vram_mb > 0 else "—"
            print(f"  {i:<4d} {job.name:<30s} {gpu:<10s} {job.cpu_weight:<8d} {job.estimated_minutes:<10d}")

        if args.max_parallel:
            print(f"\n  Mode: Legacy flat parallelism (max={args.max_parallel})")
        else:
            print(f"\n  Mode: Resource-aware")
            print(f"  GPU budget:  {args.gpu_budget_mb} MB")
            print(f"  CPU slots:   {args.cpu_slots}")

        total_est = sum(j.estimated_minutes for j in valid_jobs)
        print(f"\n  Sum of estimated runtimes: {total_est} min")
        print(f"  (Actual wall time will be much less due to parallelism)")
        print("=" * 72 + "\n")
        return

    if not valid_jobs:
        print("Nothing to run. All algorithms completed or excluded.")
        if skipped_names:
            print(f"Skipped (already completed): {', '.join(skipped_names)}")
            print("Use --no-skip to force re-run.")
        return

    # Create log directory
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = results_root / "scheduler_runs" / f"run_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up resource tracking
    if args.max_parallel:
        # Legacy mode: treat as uniform slots
        budget = ResourceBudget(gpu_budget_mb=999999, cpu_slots=args.max_parallel)
        # Override all jobs to have weight=1 and 0 GPU
        for job in valid_jobs:
            job.cpu_weight = 1
            job.gpu_vram_mb = 0
    else:
        budget = ResourceBudget(
            gpu_budget_mb=args.gpu_budget_mb,
            cpu_slots=args.cpu_slots,
        )

    print(f"\n{'=' * 72}")
    print(f"  Resource-Aware RL Baseline Scheduler")
    print(f"  Logs: {log_dir}")
    print(f"  Jobs: {len(valid_jobs)} to run, {len(skipped_names)} skipped")
    print(f"  {budget.status()}")
    print(f"{'=' * 72}\n")

    # Graceful shutdown on Ctrl+C
    shutdown_requested = False

    def handle_sigint(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\nForce killing all jobs...")
            for _, proc, handle, _, _ in running:
                proc.kill()
                handle.close()
            sys.exit(1)
        shutdown_requested = True
        print("\n\n⚠️  Shutdown requested. Waiting for running jobs to finish...")
        print("   Press Ctrl+C again to force kill.\n")

    signal.signal(signal.SIGINT, handle_sigint)

    # --- Main scheduling loop ---
    queue = list(valid_jobs)
    running = []        # [(job, proc, handle, log_file, start_time)]
    completed = []      # [(name, exit_code_str, status, duration_seconds)]
    start_time = time.time()

    while (queue or running) and not shutdown_requested:
        # Try to launch queued jobs that fit the budget
        launched_any = True
        while launched_any and queue:
            launched_any = False
            for i, job in enumerate(queue):
                if budget.can_fit(job):
                    # Launch this job
                    queue.pop(i)
                    log_file = log_dir / f"{job.name}.log"
                    handle = open(log_file, "w", encoding="utf-8")

                    cmd = [sys.executable, str(job.script_path)]
                    if job.extra_args:
                        cmd.extend(job.extra_args)

                    # Environment setup for GPU isolation and unbuffered output
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    if job.gpu_vram_mb > 0:
                        env["CUDA_VISIBLE_DEVICES"] = "0"
                        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    else:
                        # CPU-only jobs should not see the GPU at all
                        env["CUDA_VISIBLE_DEVICES"] = ""

                    gpu_tag = f"GPU:{job.gpu_vram_mb}MB" if job.gpu_vram_mb > 0 else "CPU-only"
                    print(f"  [START] {job.name:<25s} ({gpu_tag}, cpu_wt={job.cpu_weight})")

                    proc = subprocess.Popen(
                        cmd,
                        cwd=repo_root,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    budget.allocate(job)
                    running.append((job, proc, handle, log_file, time.time()))
                    launched_any = True
                    break  # restart scan from the beginning of queue

        if not running:
            if queue and not shutdown_requested:
                # All queued jobs are too big to fit — wait for something to finish
                time.sleep(args.poll_seconds)
            continue

        time.sleep(args.poll_seconds)

        # Check for finished jobs
        still_running = []
        for job, proc, handle, log_file, job_start in running:
            code = proc.poll()
            if code is None:
                still_running.append((job, proc, handle, log_file, job_start))
            else:
                handle.close()
                budget.release(job)
                duration = time.time() - job_start
                status = "ok" if code == 0 else "failed"
                code_str = str(code)
                completed.append((job.name, code_str, status, duration))
                icon = "✅" if status == "ok" else "❌"
                print(
                    f"  [{icon} DONE] {job.name:<25s} "
                    f"exit={code}  time={format_duration(duration)}  "
                    f"log={log_file}"
                )
        running = still_running

        # Show periodic dashboard
        if running and int(time.time() - start_time) % 60 < args.poll_seconds:
            print_dashboard(running, queue, completed, skipped_names, budget, start_time)

    # Handle graceful shutdown
    if shutdown_requested and running:
        print("\nWaiting for running jobs to finish...")
        for job, proc, handle, log_file, job_start in running:
            proc.wait()
            handle.close()
            duration = time.time() - job_start
            code = proc.returncode
            status = "ok" if code == 0 else "failed"
            completed.append((job.name, str(code), status, duration))
            print(f"  [DONE] {job.name} exit={code} time={format_duration(duration)}")

    # --- Summary report ---
    total_elapsed = time.time() - start_time
    ok = sum(1 for _, _, s, _ in completed if s == "ok")
    failed = sum(1 for _, _, s, _ in completed if s == "failed")
    cancelled = len(queue)  # jobs that never started due to shutdown

    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Total wall time:  {format_duration(total_elapsed)}")
    print(f"  Completed:        {ok} ok, {failed} failed")
    if skipped_names:
        print(f"  Skipped:          {len(skipped_names)} ({', '.join(skipped_names)})")
    if cancelled:
        cancelled_names = [j.name for j in queue]
        print(f"  Cancelled:        {cancelled} ({', '.join(cancelled_names)})")
    print(f"  Logs:             {log_dir}")

    if completed:
        print(f"\n  {'Algorithm':<30s} {'Status':<10s} {'Exit':<6s} {'Duration':<12s}")
        print(f"  {'-'*30} {'-'*10} {'-'*6} {'-'*12}")
        for name, code, status, duration in completed:
            icon = "✓" if status == "ok" else "✗"
            print(f"  {icon} {name:<28s} {status:<10s} {code:<6s} {format_duration(duration)}")

    # Compute theoretical sequential time vs actual
    sum_durations = sum(d for _, _, _, d in completed)
    if sum_durations > 0 and total_elapsed > 0:
        speedup = sum_durations / total_elapsed
        print(f"\n  Parallelism speedup: {speedup:.1f}x "
              f"(sum of job times: {format_duration(sum_durations)} vs wall: {format_duration(total_elapsed)})")

    print("=" * 72 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
