"""
ALFWorld benchmark: the interface the evolution loop and the reporting code
call into.
"""
from src.benchmarks.alfworld.alfworld_archive import add_node, init_archive
from src.benchmarks.alfworld.alfworld_briefing import build_mining_briefing, compare_results
from src.benchmarks.alfworld.alfworld_env import TEST_SPLIT, VALIDATION_SPLIT, games_by_type
from src.benchmarks.alfworld.alfworld_mining import (
    MINING_N, build_train_pool, mine_error_samples, mine_error_samples_from_validation,
)
from src.benchmarks.alfworld.evaluate_alfworld_predictions import evaluate_predictions
from src.benchmarks.alfworld.generate_alfworld_predictions import (
    collect_game_files, run_alfworld_episodes as run_episodes,
)
from src.benchmarks.alfworld.genome import (
    AGENT_FILE, GENOME_FILES, POLICY_FILE, SOLUTION_PROPOSER_DOMAIN_NOTES, build_root_genome,
)
from src.eval.lineage_archive import (
    FIRST_STAGE_NAME, TIER_DOUBLE_THRESHOLD, TrainPoolExhaustedError,
    assert_archive_matches_active_models, get_node, highest_completed_stage,
    increment_children_count, load_archive, new_games_for_stage, pick_best_node,
    record_stage_result, save_archive, select_parent, set_node_status, stage_plan_names,
    tier_index,
)


def test_task_pool() -> list:
    """ALFWorld's held-out test split, used for the final root-vs-best comparison."""
    pool = games_by_type(collect_game_files(TEST_SPLIT))
    return [g for games in pool.values() for g in games]
