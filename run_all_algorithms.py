import argparse
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Job:
    name: str
    script_path: Path
    extra_args: Optional[List[str]] = None


def build_jobs(repo_root: Path) -> List[Job]:
    return [
        Job("actor_critic", repo_root / "baselines/model-free/actor-critic/actor_critic.py"),
        Job("actor_critic_continuous", repo_root / "baselines/model-free/actor-critic/actor_critic_continuous.py"),
        Job("dqn", repo_root / "baselines/model-free/DQN/dqn.py"),
        Job("ddqn", repo_root / "baselines/model-free/DDQN/double_dqn.py"),
        Job("per_ddqn", repo_root / "baselines/model-free/PER/per_ddqn.py"),
        Job("dyna_q", repo_root / "baselines/model-based/Dyna-Q/dyna_q.py"),
        Job("mpc", repo_root / "baselines/model-based/MPC/learned_dynamics_mpc.py"),
        Job("muzero", repo_root / "baselines/model-based/MuZero/muzero.py"),
        Job("dreamer_v1", repo_root / "baselines/model-based/Dreamer-v1/dreamer_v1.py"),
        Job("dreamer_v2", repo_root / "baselines/model-based/Dreamer-v2/dreamer_v2.py"),
        Job("dreamer_v3", repo_root / "baselines/model-based/Dreamer-v3/dreamer_v3.py"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run all algorithms with bounded parallelism.")
    parser.add_argument("--max-parallel", type=int, default=3, help="Maximum number of concurrent jobs.")
    parser.add_argument("--poll-seconds", type=int, default=5, help="Polling interval for job completion.")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    repo_root = Path(__file__).resolve().parent
    jobs = deque(build_jobs(repo_root))

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = repo_root / "results" / "scheduler_runs" / f"run_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting scheduler with max {args.max_parallel} parallel jobs.")
    print(f"Logs: {log_dir}")

    running = []
    completed = []

    while jobs or running:
        while jobs and len(running) < args.max_parallel:
            job = jobs.popleft()
            if not job.script_path.exists():
                print(f"[SKIP] {job.name}: script not found: {job.script_path}")
                completed.append((job.name, None, "missing_script"))
                continue

            log_file = log_dir / f"{job.name}.log"
            handle = open(log_file, "w", encoding="utf-8")
            cmd = [sys.executable, str(job.script_path)]
            if job.extra_args:
                cmd.extend(job.extra_args)

            print(f"[START] {job.name} -> {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((job, proc, handle, log_file))

        if not running:
            continue

        time.sleep(args.poll_seconds)

        still_running = []
        for job, proc, handle, log_file in running:
            code = proc.poll()
            if code is None:
                still_running.append((job, proc, handle, log_file))
            else:
                handle.close()
                status = "ok" if code == 0 else "failed"
                print(f"[DONE] {job.name} exit={code} log={log_file}")
                completed.append((job.name, code, status))
        running = still_running

    ok = sum(1 for _, _, s in completed if s == "ok")
    failed = sum(1 for _, _, s in completed if s == "failed")
    skipped = sum(1 for _, _, s in completed if s == "missing_script")
    print(f"Scheduler finished. ok={ok}, failed={failed}, skipped={skipped}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
