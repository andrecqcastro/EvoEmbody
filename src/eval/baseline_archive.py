"""
Saved baselines: an evaluation of the unmutated root agent, stored per
benchmark and model pair and shared by every run with --use_baseline.

The task model samples with temperature, so a single bootstrap is a noisy
estimate. Sharing one baseline keeps runs with the same models comparable, and
--baseline_repeats averages several bootstraps.
"""
import json
import os
import re


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def baseline_path(project_root: str, benchmark_name: str, models: dict) -> str:
    key = f"{models.get('meta_model')}__{models.get('task_model')}"
    return os.path.join(project_root, "baselines", benchmark_name, f"{_sanitize(key)}.json")


def load_baseline(project_root: str, benchmark_name: str, models: dict):
    """Returns the baseline dict, or None if this (benchmark, models) pair has none yet."""
    path = baseline_path(project_root, benchmark_name, models)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(project_root: str, benchmark_name: str, models: dict, stages: dict,
                  repeats: int, repeat_results: list):
    from datetime import datetime

    path = baseline_path(project_root, benchmark_name, models)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "benchmark": benchmark_name,
        "models": models,
        "established_at": datetime.now().isoformat(),
        "repeats": repeats,
        "stages": stages,
        "repeat_results": repeat_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _summed_by_task_type(per_repeat_by_type: list) -> dict:
    """Sums {"passed", "total"} counts per task type across repeats."""
    totals = {}
    for by_type in per_repeat_by_type:
        for task_type, cell in by_type.items():
            passed, total = totals.setdefault(task_type, [0, 0])
            totals[task_type][0] = passed + cell["passed"]
            totals[task_type][1] = total + cell["total"]
    return {t: {"passed": p, "total": n} for t, (p, n) in totals.items()}


def aggregate_stage_results(stage_name: str, per_repeat_results: list) -> dict:
    """
    Averages one stage's results across repeats.

    Takes stored node["stages"][stage_name] dicts (keyed "n_tasks") and returns
    a dict in the evaluate_predictions shape (keyed "total") that
    record_stage_result accepts. pass_rate and avg_step_efficiency are averaged,
    `passed` is rounded from the mean pass rate, and by_task_type is summed.
    """
    n_tasks = per_repeat_results[0]["n_tasks"]
    mean_pass_rate = sum(r["pass_rate"] for r in per_repeat_results) / len(per_repeat_results)
    efficiencies = [r["avg_step_efficiency"] for r in per_repeat_results
                    if r.get("avg_step_efficiency") is not None]
    return {
        "total": n_tasks,
        "passed": round(mean_pass_rate * n_tasks),
        "pass_rate": mean_pass_rate,
        "avg_step_efficiency": (sum(efficiencies) / len(efficiencies)) if efficiencies else None,
        "by_task_type": _summed_by_task_type([r["by_task_type"] for r in per_repeat_results]),
    }
