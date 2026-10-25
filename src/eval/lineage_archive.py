"""
Lineage archive: every agent version produced by a run, and parent selection.

The next parent is sampled with weight

    W = sigmoid(score) * 1 / (1 + children) * B_step * B_tier

so exploration is spread across the archive instead of following a single
chain. The archive is the run's checkpoint: it is rewritten after every stage
and every child, and a resumed run continues from it. Benchmark-specific parts
(task types, curriculum construction) live in each benchmark's own
<name>_archive.py.
"""
import json
import math
import os
import random
from datetime import datetime

FIRST_STAGE_NAME = "check5"
FIRST_STAGE_SIZE = 5
# From the first non-check5 tier onward, reaching this cumulative pass rate
# doubles the tier.
TIER_DOUBLE_THRESHOLD = 0.95

STEPS_BONUS_FLOOR = 0.1
TIER_BONUS_FLOOR = 0.1

# Statuses excluded from parent selection, matched by exact value or prefix.
_INELIGIBLE_STATUSES = ("train_pool_exhausted",)
_INELIGIBLE_PREFIXES = ("discarded",)


class TrainPoolExhaustedError(RuntimeError):
    """
    Raised when mining draws every task in the pool without finding enough
    failures. The parent is then marked train_pool_exhausted and no longer selected.
    """


def build_stage_plan(total_pool_size: int, first_stage_size: int = FIRST_STAGE_SIZE,
                     tier_start_size: int = 30) -> list:
    """
    [[name, cumulative_size], ...]: check5, then tiers starting at
    tier_start_size and doubling, with the last tier covering the whole pool.

    Built from lists so a fresh archive and one loaded from JSON have the same shape.
    """
    plan = [[FIRST_STAGE_NAME, min(first_stage_size, total_pool_size)]]
    size = tier_start_size
    while True:
        if size >= total_pool_size:
            if total_pool_size > plan[-1][1]:
                plan.append([f"tier{total_pool_size}", total_pool_size])
            break
        plan.append([f"tier{size}", size])
        size *= 2
    return plan


def stage_plan_names(archive: dict) -> list:
    return [name for name, _ in archive["game_pool"]["stage_plan"]]


def new_games_for_stage(archive: dict, stage_name: str) -> list:
    """Only the games this stage adds; earlier stages' results are reused."""
    subsets = archive["game_pool"]["stage_subsets"]
    names = stage_plan_names(archive)
    position = names.index(stage_name)
    if position == 0:
        return list(subsets[stage_name])
    return subsets[stage_name][len(subsets[names[position - 1]]):]


def load_archive(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_archive(archive: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)


def assert_archive_matches_active_models(archive: dict, archive_path: str):
    """Refuses to continue a lineage whose scores were produced by other models."""
    from src.config.llm_config import describe_active_models

    active = describe_active_models()
    stored = archive.get("models")
    if stored != active:
        raise RuntimeError(
            f"Model mismatch for archive '{archive_path}'.\n"
            f"  archive was built with : {stored}\n"
            f"  current configuration  : {active}\n"
            f"Continuing would rank agents evaluated under different models against each "
            f"other. Start a fresh run (drop --resume) for this configuration.")


def get_node(archive: dict, node_id: str) -> dict:
    for node in archive["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"Node not found: {node_id}")


def add_node(archive: dict, parent_id, agent_json_path: str, cycle_index: int,
            id_prefix: str = "v") -> str:
    node_id = f"{id_prefix}{archive['next_id']}"
    archive["next_id"] += 1
    archive["nodes"].append({
        "id": node_id,
        "parent_id": parent_id,
        "agent_json_path": agent_json_path,
        "cycle_index": cycle_index,
        "created_at": datetime.now().isoformat(),
        "status": "alive",
        "score": None,
        "avg_step_efficiency": None,
        "children_count": 0,
        "children_ids": [],
        "no_think_fallbacks": 0,
        "mutation": {},
        "stages": {},
        # Reporting-only snapshots of the selection weight terms.
        "bonus_sources": {},
        "bonus_values": {},
    })
    if parent_id is not None:
        get_node(archive, parent_id)["children_ids"].append(node_id)
    return node_id


def increment_children_count(archive: dict, parent_id, steps_bonus_mode: str = "raw"):
    if parent_id is not None:
        node = get_node(archive, parent_id)
        node["children_count"] += 1
        _record_bonus_sources(archive, node, steps_bonus_mode)


def record_stage_result(archive: dict, node_id: str, stage_name: str, results: dict,
                        predictions_path: str, eval_log_path: str, results_json_path: str,
                        steps_bonus_mode: str = "raw"):
    node = get_node(archive, node_id)
    node["stages"][stage_name] = {
        "n_tasks": results["total"],
        "passed": results["passed"],
        "pass_rate": results["pass_rate"],
        "avg_step_efficiency": results["avg_step_efficiency"],
        "by_task_type": results["by_task_type"],
        "predictions_path": predictions_path,
        "eval_log_path": eval_log_path,
        "results_json_path": results_json_path,
        "evaluated_at": datetime.now().isoformat(),
    }
    node["score"] = results["pass_rate"]
    node["avg_step_efficiency"] = results["avg_step_efficiency"]
    _record_bonus_sources(archive, node, steps_bonus_mode)


def set_node_status(archive: dict, node_id: str, status: str):
    get_node(archive, node_id)["status"] = status


def highest_completed_stage(node: dict):
    """The highest stage reached (stages are recorded in plan order)."""
    return next(reversed(node["stages"]), None)


def tier_index(archive: dict, node: dict) -> int:
    """Position of a node's highest completed stage within the archive's
    stage_plan, or -1 if it has not completed any stage yet."""
    stage = highest_completed_stage(node)
    if stage is None:
        return -1
    names = stage_plan_names(archive)
    return names.index(stage) if stage in names else -1


def compute_steps_bonus_factor(avg_step_efficiency, mode: str = "raw") -> float:
    if avg_step_efficiency is None:
        return STEPS_BONUS_FLOOR
    if mode == "raw":
        return max(STEPS_BONUS_FLOOR, avg_step_efficiency)
    if mode == "sigmoid":
        return 1 / (1 + math.exp(-10 * (avg_step_efficiency - 0.5)))
    raise ValueError(f"unknown steps_bonus_mode: {mode!r}")


def compute_tier_bonus_factor(tier_idx: int, total_tiers: int) -> float:
    """Grows with progress through the stage plan, with a floor."""
    if tier_idx < 0 or total_tiers <= 0:
        return TIER_BONUS_FLOOR
    return max(TIER_BONUS_FLOOR, (tier_idx + 1) / total_tiers)


def _record_bonus_sources(archive: dict, node: dict, steps_bonus_mode: str = "raw") -> None:
    """
    Stores the terms of the selection weight on the node for reporting.
    Selection always recomputes them in _node_weight.
    """
    total_tiers = len(archive["game_pool"]["stage_plan"])
    idx = tier_index(archive, node)
    node["bonus_sources"] = {
        "tier_index": idx,
        "children_count": node.get("children_count", 0),
        "avg_step_efficiency": node.get("avg_step_efficiency"),
    }
    node["bonus_values"] = {
        "tier_bonus": compute_tier_bonus_factor(idx, total_tiers),
        "steps_bonus": compute_steps_bonus_factor(node.get("avg_step_efficiency"), steps_bonus_mode),
        "children_factor": 1 / (1 + node.get("children_count", 0)),
    }


def _node_weight(archive: dict, node: dict, steps_bonus_mode: str) -> float:
    score = node.get("score") or 0.0
    sigmoid_score = 1 / (1 + math.exp(-10 * (score - 0.5)))
    children_factor = 1 / (1 + node.get("children_count", 0))
    steps_bonus = compute_steps_bonus_factor(node.get("avg_step_efficiency"), steps_bonus_mode)
    total_tiers = len(archive["game_pool"]["stage_plan"])
    tier_bonus = compute_tier_bonus_factor(tier_index(archive, node), total_tiers)
    return sigmoid_score * children_factor * steps_bonus * tier_bonus


def pick_best_node(nodes: list) -> dict:
    """Highest score, ties broken by the highest avg_step_efficiency."""
    best_score = max(n["score"] for n in nodes)
    tied = [n for n in nodes if n["score"] == best_score]
    return max(tied, key=lambda n: n.get("avg_step_efficiency") or 0)


def _is_eligible(node: dict) -> bool:
    status = node["status"]
    return status not in _INELIGIBLE_STATUSES and not status.startswith(_INELIGIBLE_PREFIXES)


def select_parent(archive: dict, steps_bonus_mode: str = "raw", rng=None,
                  parent_selection: str = "weighted") -> str:
    eligible = [n for n in archive["nodes"] if _is_eligible(n)]
    if not eligible:
        raise ValueError("No eligible nodes in the archive for parent selection.")

    if parent_selection == "greedy":
        return pick_best_node(eligible)["id"]

    weights = [_node_weight(archive, n, steps_bonus_mode) for n in eligible]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(eligible)
    return (rng or random).choices(eligible, weights=weights, k=1)[0]["id"]
