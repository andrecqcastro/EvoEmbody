"""
ALFWorld validation curriculum, built on the generic lineage mechanics in
src/eval/lineage_archive.py.
"""
from datetime import datetime

from src.benchmarks.alfworld.alfworld_env import ALFWORLD_TASK_TYPES, TASK_TYPE_OPTIMAL_STEPS
from src.eval import lineage_archive
from src.eval.lineage_archive import FIRST_STAGE_SIZE, build_stage_plan

# All 6 task types, easiest (fewest optimal steps) first; ties broken by name.
TYPE_ORDER = tuple(sorted(ALFWORLD_TASK_TYPES, key=lambda t: (TASK_TYPE_OPTIMAL_STEPS[t], t)))


def add_node(archive: dict, parent_id, agent_json_path: str, cycle_index: int) -> str:
    return lineage_archive.add_node(archive, parent_id, agent_json_path, cycle_index,
                                    id_prefix="alfworld_v")


def build_curriculum_subsets(pool_by_type: dict, stage_plan: list) -> dict:
    """
    Builds one growing sequence of game files, round-robining across TYPE_ORDER.
    Prefixes of that sequence are both nested (an earlier stage is a subset of
    every later one) and type-balanced by construction.
    """
    active = [t for t in TYPE_ORDER if t in pool_by_type]
    if not active:
        raise ValueError("No task types with available games.")

    queues = {t: sorted(paths) for t, paths in pool_by_type.items()}
    index = {t: 0 for t in queues}
    sequence, subsets = [], {}

    target_total = stage_plan[-1][1]
    position = exhausted = 0
    while len(sequence) < target_total:
        task_type = active[position % len(active)]
        position += 1
        if index[task_type] >= len(queues[task_type]):
            exhausted += 1
            if exhausted > len(active):
                raise ValueError(
                    f"Not enough games to build the curriculum: {len(sequence)}/{target_total} "
                    f"collected, pool exhausted for {active}")
            continue
        sequence.append(queues[task_type][index[task_type]])
        index[task_type] += 1
        exhausted = 0

    for stage_name, size in stage_plan:
        subsets[stage_name] = list(sequence[:size])

    return subsets


def init_archive(split: str, first_stage_size: int = FIRST_STAGE_SIZE, archive_path: str = None) -> dict:
    from src.config.llm_config import describe_active_models
    from src.benchmarks.alfworld.alfworld_env import games_by_type
    from src.benchmarks.alfworld.generate_alfworld_predictions import collect_game_files

    pool_by_type = games_by_type(collect_game_files(split))
    missing = [t for t in TYPE_ORDER if t not in pool_by_type]
    if missing:
        raise ValueError(f"No games found for task types: {missing}")

    total_pool_size = sum(len(v) for v in pool_by_type.values())
    # check5 -> tier30 -> tier60 -> tier120 -> full pool.
    stage_plan = build_stage_plan(total_pool_size, first_stage_size)

    archive = {
        "schema_version": 3,
        "benchmark": "alfworld",
        "created_at": datetime.now().isoformat(),
        # Scores are only comparable within one model configuration.
        "models": describe_active_models(),
        "next_id": 0,
        # The root's check5 pass count, set after bootstrap.
        "check5_gate": None,
        "game_pool": {
            "split": split,
            "task_types": list(TYPE_ORDER),
            "stage_plan": stage_plan,
            "per_type_counts": {t: len(v) for t, v in pool_by_type.items()},
            "total_pool_size": total_pool_size,
            "stage_subsets": build_curriculum_subsets(pool_by_type, stage_plan),
        },
        "nodes": [],
    }
    if archive_path:
        lineage_archive.save_archive(archive, archive_path)
    return archive
