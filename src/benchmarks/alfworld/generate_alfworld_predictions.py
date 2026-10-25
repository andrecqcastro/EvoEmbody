"""
Plays ALFWorld episodes with one genome and writes a predictions JSONL.

Each episode runs in its own process, because TextWorld keeps module-level
state that breaks when several games are stepped in one interpreter. Each
process writes its turn-by-turn log to logs/episodes/ and returns only the
episode record.
"""
import contextlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.benchmarks.alfworld.alfworld_env import (
    extract_task_instruction, get_episode_max_steps, get_task_type_from_gamefile, is_success,
)
from src.benchmarks.alfworld.genome import load_genome
from src.utils.run_context import record_fallback

ALFWORLD_CONFIG = os.path.join(_PROJECT_ROOT, "alfworld_config.yaml")


def collect_game_files(split: str) -> list:
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    with open(ALFWORLD_CONFIG) as f:
        config = yaml.safe_load(f)
    for key in ("data_path", "eval_id_data_path", "eval_ood_data_path"):
        config["dataset"][key] = os.path.expandvars(config["dataset"][key])
    return AlfredTWEnv(config, train_eval=split).game_files


def _play_episode(game_file: str, episode_idx: int, split: str, agent_json_path: str,
                  log_path: str) -> tuple:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from src.config.llm_config import get_task_llm

    ArchiveAgent, policy = load_genome(agent_json_path)
    task_type = get_task_type_from_gamefile(game_file)
    max_steps = get_episode_max_steps(task_type)

    llm = get_task_llm()
    tokens_in = tokens_out = 0
    inner_invoke = llm.invoke

    def tracked_invoke(*args, **kwargs):
        nonlocal tokens_in, tokens_out
        result = inner_invoke(*args, **kwargs)
        usage = getattr(result, "usage_metadata", None)
        if usage:
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
        return result

    llm.invoke = tracked_invoke
    agent = ArchiveAgent(llm=llm)
    # extract_thought is optional: a mutation may remove it.
    extract_thought = getattr(policy, "extract_thought", None)

    env_id = textworld.gym.register_games(
        [game_file],
        textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"]),
        batch_size=1, asynchronous=False, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(), AlfredInfos],
    )

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        # Everything the genome prints goes to the episode log.
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            env = textworld.gym.make(env_id)
            obs_list, infos = env.reset()
            obs = obs_list[0]

            gamefile = infos.get("extra.gamefile", [game_file])
            gamefile = gamefile[0] if isinstance(gamefile, list) else gamefile
            instruction = extract_task_instruction(obs) or f"task_{episode_idx}"

            print(f"Episode {episode_idx} [{task_type}] {instruction} (max {max_steps} steps)")
            print(f"game: {gamefile}\n")

            history = [{"turn": 0, "obs": obs, "action": None, "thought": "", "done": False}]
            success = False

            for turn in range(1, max_steps + 1):
                commands = infos.get("admissible_commands", [[]])
                commands = commands[0] if commands and isinstance(commands[0], list) else []
                prompt = policy.build_alfworld_prompt(instruction, history, commands)
                try:
                    response = agent.generate_response(prompt)
                except Exception as e:
                    print(f"agent error on turn {turn}: {e}")
                    record_fallback("episode_agent_error", episode=episode_idx,
                                    turn=turn, error=str(e)[:300])
                    break

                action = policy.parse_alfworld_action(response)
                thought = extract_thought(response) if callable(extract_thought) else ""
                if not action:
                    record_fallback("episode_empty_action", episode=episode_idx, turn=turn)

                obs_list, _scores, dones, infos = env.step([action])
                obs = obs_list[0]
                won = infos.get("won", [False])
                won = won[0] if isinstance(won, list) else won
                success = bool(won) or is_success(obs)

                history.append({"turn": turn, "obs": obs, "action": action,
                                "thought": thought, "done": dones[0]})
                print(f"[turn {turn}] > {action}\n{obs}\n")

                if success or dones[0]:
                    break

            env.close()
            print(f"{'SUCCESS' if success else 'FAIL'} in {len(history) - 1}/{max_steps} steps")

    record = {
        "instance_id": f"alfworld_{split}_{episode_idx:03d}",
        "task_instruction": instruction,
        "task_type": task_type,
        "gamefile": gamefile,
        "success": success,
        "num_steps": len(history) - 1,
        "max_steps_used": max_steps,
        "episode_log": log_path,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "interaction_history": history,
    }
    return episode_idx, record, tokens_in, tokens_out


def run_alfworld_episodes(agent_json_path: str, game_files: list, split: str, max_workers: int,
                          results_dir: str, episodes_dir: str, tag: str):
    """
    Plays one batch of episodes, inside the genome sandbox when it is enabled
    (see src/utils/sandbox.py) and on the host otherwise.

    Returns (predictions_path, tokens_in, tokens_out).
    """
    from src.utils.sandbox import run_module, sandbox_enabled

    if not sandbox_enabled():
        return _run_alfworld_episodes_direct(agent_json_path, game_files, split, max_workers,
                                             results_dir, episodes_dir, tag)

    result = run_module(
        project_root=_PROJECT_ROOT,
        module="src.benchmarks.alfworld.run_episode_batch",
        args=["--agent-json-path", agent_json_path, "--game-files", json.dumps(game_files),
             "--split", split, "--max-workers", str(max_workers),
             "--results-dir", results_dir, "--episodes-dir", episodes_dir, "--tag", tag],
        writable_dirs=[os.path.commonpath([os.path.abspath(results_dir),
                                          os.path.abspath(episodes_dir)])],
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"sandboxed episode batch {tag} failed (exit {result['returncode']}):\n"
                           f"{result['stderr'][-4000:]}")
    payload = json.loads(result["stdout"].strip().splitlines()[-1])
    return payload["predictions_path"], payload["tokens_in"], payload["tokens_out"]


def _run_alfworld_episodes_direct(agent_json_path: str, game_files: list, split: str,
                                  max_workers: int, results_dir: str, episodes_dir: str,
                                  tag: str):
    """Episode playing without the sandbox dispatch; also what runs inside the container."""
    # Fail once here on a broken genome rather than once per episode.
    load_genome(agent_json_path)
    agent_json_path = os.path.abspath(agent_json_path)

    if not game_files:
        raise ValueError(f"No game files to play for split={split}.")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)
    workers = min(len(game_files), max_workers)
    print(f"Playing {len(game_files)} episodes on {workers} parallel workers "
          f"(per-episode logs in {episodes_dir})")

    records = {}
    tokens_in = tokens_out = 0
    # "spawn" gives each worker a clean interpreter without inherited TextWorld state.
    context = multiprocessing.get_context("spawn")
    remaining = list(enumerate(game_files))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        while remaining:
            wave, remaining = remaining[:workers], remaining[workers:]
            futures = {
                executor.submit(_play_episode, game_file, idx, split, agent_json_path,
                                os.path.join(episodes_dir, f"{tag}_ep{idx:03d}.log")): idx
                for idx, game_file in wave
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, record, episode_in, episode_out = future.result()
                    records[idx] = record
                    tokens_in += episode_in
                    tokens_out += episode_out
                except Exception as e:
                    print(f"  episode {idx} crashed: {e}")
                    record_fallback("episode_crashed", episode=idx, tag=tag, error=str(e)[:300])

    predictions_path = os.path.join(results_dir, f"predictions_{tag}.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for idx in sorted(records):
            f.write(json.dumps(records[idx], ensure_ascii=False) + "\n")

    return predictions_path, tokens_in, tokens_out
