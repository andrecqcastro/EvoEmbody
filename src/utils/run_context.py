"""
Per-run directory layout and the shared fallback log.

Every run owns one directory under runs/ (or runs/<run_tag>/ when tagged):

    runs/[<run_tag>/]run_<timestamp>/
        archive.json          lineage archive (the resumable checkpoint)
        fallbacks.jsonl       every automatic degradation, one JSON per line
        logs/                 vLLM logs, evaluation logs, meta-agent traces
        logs/episodes/        one file per ALFWorld episode (full turn-by-turn)
        eval_results/         predictions JSONL and results JSON per stage
        artifacts/            mutated agent JSONs, meta-agent reports
        test_results/         final root-vs-best replay on the held-out split
        summary.txt           final per-node summary
        evolution_graph.html  lineage graph, with the summary below it
        evolution_graph.png   the same graph on its own

Starting from scratch moves any existing runs/[<run_tag>/] aside into
previous_[<run_tag>_]<timestamp>/. Different run tags never touch each other.
"""
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

FALLBACK_LOG_ENV = "ALFWORLD_FALLBACK_LOG"


@dataclass
class RunPaths:
    project_root: str
    run_dir: str
    logs_dir: str
    episodes_dir: str
    results_dir: str
    artifacts_dir: str
    archive_path: str
    fallback_log: str
    resumed_from: str = None

    @property
    def summary_path(self) -> str:
        return os.path.join(self.run_dir, "summary.txt")

    @property
    def graph_path(self) -> str:
        return os.path.join(self.run_dir, "evolution_graph.html")

    @property
    def graph_png_path(self) -> str:
        return os.path.join(self.run_dir, "evolution_graph.png")

    def cycle_dir(self, name: str) -> str:
        path = os.path.join(self.artifacts_dir, name)
        os.makedirs(path, exist_ok=True)
        return path


def _runs_root(project_root: str, run_tag: str = None) -> str:
    return os.path.join(project_root, "runs", run_tag) if run_tag else os.path.join(project_root, "runs")


def latest_run_dir(project_root: str, run_tag: str = None):
    runs = _runs_root(project_root, run_tag)
    if not os.path.isdir(runs):
        return None
    candidates = sorted(
        os.path.join(runs, d) for d in os.listdir(runs)
        if d.startswith("run_") and os.path.exists(os.path.join(runs, d, "archive.json"))
    )
    return candidates[-1] if candidates else None


def _archive_previous(project_root: str, run_tag: str = None) -> str:
    """Moves this tag's runs aside so a from-scratch run never appends to old results."""
    runs = _runs_root(project_root, run_tag)
    if not os.path.isdir(runs) or not os.listdir(runs):
        return None
    prefix = f"{run_tag}_" if run_tag else ""
    destination = os.path.join(project_root, f"previous_{prefix}{datetime.now():%Y%m%d_%H%M%S_%f}")
    os.makedirs(destination, exist_ok=True)
    shutil.move(runs, os.path.join(destination, "runs"))
    print(f"Previous runs moved to {destination}")
    return destination


def start_run(project_root: str, resume: bool = False, run_tag: str = None) -> RunPaths:
    resumed_from = None
    if resume:
        resumed_from = latest_run_dir(project_root, run_tag)
        if resumed_from is None:
            raise SystemExit(
                "--resume was requested but no previous run with an archive was found under "
                f"{_runs_root(project_root, run_tag)}. Run without --resume to start from scratch."
            )
    else:
        _archive_previous(project_root, run_tag)

    base = os.path.join(_runs_root(project_root, run_tag), f"run_{datetime.now():%Y%m%d_%H%M%S_%f}")
    run_dir, attempt = base, 1
    while os.path.exists(run_dir):
        attempt += 1
        run_dir = f"{base}_{attempt}"
    paths = RunPaths(
        project_root=project_root,
        run_dir=run_dir,
        logs_dir=os.path.join(run_dir, "logs"),
        episodes_dir=os.path.join(run_dir, "logs", "episodes"),
        results_dir=os.path.join(run_dir, "eval_results"),
        artifacts_dir=os.path.join(run_dir, "artifacts"),
        archive_path=os.path.join(run_dir, "archive.json"),
        fallback_log=os.path.join(run_dir, "fallbacks.jsonl"),
        resumed_from=resumed_from,
    )
    for directory in (paths.logs_dir, paths.episodes_dir, paths.results_dir, paths.artifacts_dir):
        os.makedirs(directory, exist_ok=True)

    if resumed_from:
        shutil.copy(os.path.join(resumed_from, "archive.json"), paths.archive_path)
        print(f"Resuming the lineage of {resumed_from}")

    # Episode workers are separate processes and find the log through the environment.
    os.environ[FALLBACK_LOG_ENV] = paths.fallback_log
    return paths


def record_fallback(kind: str, **fields) -> None:
    """Appends one automatic degradation to the run's fallback log (process-safe append)."""
    path = os.environ.get(FALLBACK_LOG_ENV)
    if not path:
        return
    entry = {"at": datetime.now().isoformat(), "kind": kind, "pid": os.getpid(), **fields}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_fallbacks(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
