"""
Fixed facts about the ALFWorld environment and how episodes are scored.

Evaluation infrastructure, not part of the agent genome: the meta-agents never
see or modify this file. The mutable half lives in alfworld_policy.py.
"""
import os

SUCCESS_STRING = "You won!"

# Candidates are scored on the in-distribution split. The out-of-distribution
# split is held out from evolution and played once, on the final agent.
VALIDATION_SPLIT = "eval_in_distribution"
TEST_SPLIT = "eval_out_of_distribution"

ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

# Steps needed to solve each task type when the object is found on the first
# attempt. An episode is cut off at STEPS_MULTIPLIER x optimal, and step
# efficiency is optimal/actual on solved episodes.
TASK_TYPE_OPTIMAL_STEPS = {
    "pick_and_place_simple": 5,
    "look_at_obj_in_light": 5,
    "pick_clean_then_place_in_recep": 7,
    "pick_heat_then_place_in_recep": 7,
    "pick_cool_then_place_in_recep": 8,
    "pick_two_obj_and_place": 10,
}
STEPS_MULTIPLIER = 8
FALLBACK_MAX_STEPS = 50


def games_by_type(game_files: list) -> dict:
    """Buckets game files by task type, dropping anything outside the 6 known types."""
    pool = {}
    for game_file in game_files:
        task_type = get_task_type_from_gamefile(game_file)
        if task_type in ALFWORLD_TASK_TYPES:
            pool.setdefault(task_type, []).append(game_file)
    return pool


def get_episode_max_steps(task_type: str) -> int:
    optimal = TASK_TYPE_OPTIMAL_STEPS.get(task_type)
    return FALLBACK_MAX_STEPS if optimal is None else optimal * STEPS_MULTIPLIER


def is_success(observation: str) -> bool:
    return SUCCESS_STRING in observation


def count_nothing_happens(history: list) -> int:
    return sum(1 for h in history if "Nothing happens" in h.get("obs", ""))


def get_task_type_from_gamefile(gamefile: str) -> str:
    folder = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
    for task_name in sorted(ALFWORLD_TASK_TYPES, key=len, reverse=True):
        if folder == task_name or folder.startswith(task_name + "-"):
            return task_name
    return "unknown"


def extract_task_instruction(obs: str) -> str:
    for line in obs.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("your task is to"):
            return stripped.split(":", 1)[-1].strip().rstrip(".")
    return ""
