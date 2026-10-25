"""
Scores one predictions JSONL: pass rate, step efficiency and per-task-type counts.

Prints a short summary; per-episode detail goes to the evaluation log and the
results JSON.
"""
import json
import os
from datetime import datetime

from src.benchmarks.alfworld.alfworld_env import TASK_TYPE_OPTIMAL_STEPS, count_nothing_happens


def evaluate_predictions(predictions_path: str, tag: str, logs_dir: str, results_dir: str):
    """Returns (eval_log_path, results_json_path, results) or (None, None, None)."""
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    if not predictions:
        print(f"No predictions in {predictions_path}.")
        return None, None, None

    episodes = {}
    log_lines = []
    by_task_type = {}
    passed = solved = 0
    total_steps = total_nothing = 0
    total_efficiency = 0.0

    for prediction in predictions:
        instance_id = prediction["instance_id"]
        task_type = prediction.get("task_type", "unknown")
        success = prediction.get("success", False)
        num_steps = prediction.get("num_steps", 0)
        nothing_happens = count_nothing_happens(prediction.get("interaction_history", []))

        total_steps += num_steps
        total_nothing += nothing_happens

        efficiency = None
        optimal = TASK_TYPE_OPTIMAL_STEPS.get(task_type)
        if success and num_steps > 0 and optimal is not None:
            efficiency = min(1.0, optimal / num_steps)
            total_efficiency += efficiency
            solved += 1
        passed += bool(success)

        counts = by_task_type.setdefault(task_type, {"total": 0, "passed": 0})
        counts["total"] += 1
        counts["passed"] += bool(success)

        episodes[instance_id] = {
            "success": success,
            "num_steps": num_steps,
            "task_type": task_type,
            "nothing_happens_count": nothing_happens,
            "task_instruction": prediction.get("task_instruction", instance_id),
            "step_efficiency": round(efficiency, 4) if efficiency is not None else None,
            "episode_log": prediction.get("episode_log"),
        }

        log_lines += [
            f"=== {instance_id} ({'PASS' if success else 'FAIL'}) ===",
            f"Task type: {task_type}",
            f"Instruction: {prediction.get('task_instruction', '')}",
            f"Steps: {num_steps} | Nothing happens: {nothing_happens}",
            f"Episode log: {prediction.get('episode_log')}",
            "",
        ]

    total = len(predictions)
    results = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4),
        "avg_steps": round(total_steps / total, 2),
        "avg_nothing_happens": round(total_nothing / total, 2),
        "avg_step_efficiency": round(total_efficiency / solved, 4) if solved else None,
        "by_task_type": by_task_type,
        "episodes": episodes,
    }

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    eval_log_path = os.path.join(logs_dir, f"eval_{tag}.log")
    results_json_path = os.path.join(results_dir, f"results_{tag}.json")
    with open(eval_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(format_stage_summary(tag, results))
    return eval_log_path, results_json_path, results


def format_stage_summary(tag: str, results: dict) -> str:
    """One-paragraph summary of a stage: episodes run, passed, failed and the score."""
    succeeded = sorted(i for i, e in results["episodes"].items() if e["success"])
    failed = sorted(i for i, e in results["episodes"].items() if not e["success"])
    short = lambda ids: ", ".join(i.rsplit("_", 1)[-1] for i in ids) or "none"
    efficiency = results["avg_step_efficiency"]
    return "\n".join([
        f"[task agent] {tag}: evaluated {results['total']} episodes",
        f"  success ({len(succeeded)}): {short(succeeded)}",
        f"  failure ({len(failed)}): {short(failed)}",
        "  by task type: " + ", ".join(
            f"{t} {c['passed']}/{c['total']}" for t, c in sorted(results["by_task_type"].items())),
        f"  SCORE {results['pass_rate']:.1%}"
        f" | avg steps {results['avg_steps']}"
        f" | step efficiency {efficiency if efficiency is not None else 'n/a'}",
    ])
