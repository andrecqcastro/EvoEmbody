"""
Error mining: collects failures of one genome for the Problem Understanding agent.

By default failures are drawn from the training split, which is disjoint from
the validation games candidates are scored on. Games are played one at a time,
each in a fresh process, until n episodes fail or the pool is exhausted.
"""
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor

from src.benchmarks.alfworld.generate_alfworld_predictions import _PROJECT_ROOT, _play_episode
from src.benchmarks.alfworld.genome import load_genome
from src.eval.lineage_archive import TrainPoolExhaustedError, highest_completed_stage

# Failures handed to Problem Understanding per cycle.
MINING_N = 1


def build_train_pool() -> list:
    """The flat pool of train-split games across the known task types, drawn from once per run."""
    from src.benchmarks.alfworld.alfworld_env import games_by_type
    from src.benchmarks.alfworld.generate_alfworld_predictions import collect_game_files

    pool_by_type = games_by_type(collect_game_files("train"))
    return [g for games in pool_by_type.values() for g in games]


def _mine_loop(pool: list, n: int, episodes_dir: str, tag: str, play) -> tuple:
    """
    Plays games from the pool until n failures are found.

    play(game_file, episode_idx, log_path) -> (record, tokens_in, tokens_out).
    Returns (failures, attempted, tokens_in, tokens_out).
    """
    failures, attempted = [], 0
    tokens_in = tokens_out = 0
    for game_file in pool:
        if len(failures) >= n:
            break
        log_path = os.path.join(episodes_dir, f"{tag}_mine{attempted:03d}.log")
        record, ep_in, ep_out = play(game_file, attempted, log_path)
        attempted += 1
        tokens_in += ep_in
        tokens_out += ep_out
        if not record["success"]:
            failures.append(record)
    return failures, attempted, tokens_in, tokens_out


def mine_error_samples(agent_json_path: str, train_pool: list, split: str, n: int,
                       results_dir: str, episodes_dir: str, tag: str) -> dict:
    """
    Draws training-split failures, inside the genome sandbox when it is enabled
    (see src/utils/sandbox.py) and on the host otherwise.
    """
    from src.utils.sandbox import run_module, sandbox_enabled

    if not sandbox_enabled():
        return _mine_error_samples_direct(agent_json_path, train_pool, split, n,
                                          results_dir, episodes_dir, tag)

    result = run_module(
        project_root=_PROJECT_ROOT,
        module="src.benchmarks.alfworld.run_mining_batch",
        args=["--agent-json-path", agent_json_path, "--train-pool", json.dumps(train_pool),
             "--split", split, "--n", str(n), "--results-dir", results_dir,
             "--episodes-dir", episodes_dir, "--tag", tag],
        writable_dirs=[os.path.commonpath([os.path.abspath(results_dir),
                                          os.path.abspath(episodes_dir)])],
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"sandboxed mining batch {tag} failed (exit {result['returncode']}):\n"
                           f"{result['stderr'][-4000:]}")
    return json.loads(result["stdout"].strip().splitlines()[-1])


def _mine_error_samples_direct(agent_json_path: str, train_pool: list, split: str, n: int,
                               results_dir: str, episodes_dir: str, tag: str) -> dict:
    """
    Mining without the sandbox dispatch; also what runs inside the container.

    Returns {"samples_path", "attempted", "failure_ids", "tokens_in", "tokens_out", ...}.
    """
    load_genome(agent_json_path)
    agent_json_path = os.path.abspath(agent_json_path)

    pool = list(train_pool)
    random.shuffle(pool)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        def play(game_file, idx, log_path):
            _, record, ep_in, ep_out = executor.submit(
                _play_episode, game_file, idx, split, agent_json_path, log_path).result()
            return record, ep_in, ep_out

        failures, attempted, tokens_in, tokens_out = _mine_loop(pool, n, episodes_dir, tag, play)

    if len(failures) < n:
        raise TrainPoolExhaustedError(
            f"mine_error_samples exhausted the {len(pool)}-game training pool for {tag} and "
            f"only found {len(failures)}/{n} failures.")

    samples_path = os.path.join(results_dir, f"error_samples_{tag}.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for record in failures:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "samples_path": samples_path,
        "attempted": attempted,
        "failure_ids": [r["gamefile"] for r in failures],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "source": "train",
        "n_found": len(failures),
    }


def mine_error_samples_from_validation(parent: dict, n: int, results_dir: str, tag: str) -> dict:
    """
    Validation-pool mining (--mining_source validation): reuses the failures the
    parent already produced on its highest validation stage. No new episodes are
    played. Returns the same shape as mine_error_samples.
    """
    stage = highest_completed_stage(parent)
    if stage is None:
        raise RuntimeError(
            f"{parent['id']} has not completed any validation stage, so there is nothing "
            f"to mine under --mining_source validation.")

    stage_record = parent["stages"][stage]
    predictions_path = stage_record["predictions_path"]
    with open(predictions_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    failures = [r for r in records if not r.get("success")][:n]

    os.makedirs(results_dir, exist_ok=True)
    samples_path = os.path.join(results_dir, f"error_samples_{tag}.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for record in failures:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "samples_path": samples_path,
        "attempted": len(records),
        "failure_ids": [r["gamefile"] for r in failures],
        "tokens_in": 0,
        "tokens_out": 0,
        "source": "validation",
        "n_found": len(failures),
        "eval_log_path": stage_record["eval_log_path"],
    }
