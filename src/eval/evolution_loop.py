"""
The evolution loop.

Per cycle: select a parent from the archive, mine its failures (from the
training split, or from its own validation episodes under --mining_source
validation), run the three meta-agents to diagnose, plan and apply a mutation,
then evaluate the child through the validation tiers (check5 -> tier30 ->
tier60 -> ...) with a gate after each. The archive is saved after every stage,
so an interrupted run resumes where it stopped.
"""
import json
import os

from src.agents.implementation_agent import ImplementationAgent
from src.agents.meta_loop import ContextLengthError
from src.agents.problem_understanding_agent import ProblemUnderstandingAgent
from src.agents.solution_proposer_agent import SolutionProposerAgent
from src.benchmarks.registry import get_benchmark
from src.eval.lineage_archive import TrainPoolExhaustedError
from src.utils.run_context import read_fallbacks
from src.utils.run_report import print_meta_summary, summarize_problem, summarize_solution, write_final_report

# --max_tokens stops new cycles from starting once the budget is reached. The
# cycle in progress may run this far past the budget to finish; it bounds each
# meta-agent's tool loop.
MAX_CYCLE_OVERAGE_TOKENS = 10_000_000


def _merge_predictions(prior_path, new_path: str, results_dir: str, tag: str) -> str:
    """
    Concatenates the previous stage's episodes with this stage's new ones and
    renumbers instance_id over the merged set.
    """
    records = []
    for path in (prior_path, new_path):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                records += [json.loads(line) for line in f if line.strip()]

    for i, record in enumerate(records):
        record["instance_id"] = f"{record['instance_id'].rsplit('_', 1)[0]}_{i:03d}"

    merged_path = os.path.join(results_dir, f"predictions_{tag}_merged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return merged_path


def _stage_status(benchmark, stage_name: str, position: int, total_stages: int, results: dict,
                  is_root: bool, check5_gate) -> str:
    """
    The gate decision for one stage's cumulative result. Returns "" when the node
    should continue to the next stage.
    """
    if stage_name == benchmark.FIRST_STAGE_NAME:
        if not is_root and results["passed"] < check5_gate:
            return f"discarded_{stage_name}"
    elif results["pass_rate"] < benchmark.TIER_DOUBLE_THRESHOLD:
        return f"capped_{stage_name}"
    if position == total_stages - 1:
        return f"reached_{stage_name}"
    return ""


def _evaluate_node(benchmark, agent_json_path: str, node_id: str, archive: dict, paths,
                   max_workers: int, is_root: bool = False, steps_bonus_mode: str = "raw"):
    """
    Runs check5 -> gate -> tier30 -> gate -> tier60 -> ... Each stage only plays
    the games it adds to the previous one.

    Returns (status, tokens_in, tokens_out).
    """
    total_in = total_out = 0
    prior_merged = None
    stage_plan = archive["game_pool"]["stage_plan"]

    for position, (stage_name, _size) in enumerate(stage_plan):
        tag = f"{node_id}_{stage_name}"
        predictions_path, tokens_in, tokens_out = benchmark.run_episodes(
            agent_json_path=agent_json_path,
            game_files=benchmark.new_games_for_stage(archive, stage_name),
            split=benchmark.VALIDATION_SPLIT, max_workers=max_workers,
            results_dir=paths.results_dir, episodes_dir=paths.episodes_dir, tag=tag,
        )
        total_in += tokens_in
        total_out += tokens_out

        merged_path = _merge_predictions(prior_merged, predictions_path, paths.results_dir, tag)
        eval_log_path, results_json_path, results = benchmark.evaluate_predictions(
            merged_path, tag, paths.logs_dir, paths.results_dir)

        if results is None:
            raise RuntimeError(
                f"No episode of {node_id} at {stage_name} produced a result "
                f"({merged_path}). Check {paths.episodes_dir} before retrying.")

        benchmark.record_stage_result(archive, node_id, stage_name, results,
                                      merged_path, eval_log_path, results_json_path,
                                      steps_bonus_mode)

        status = _stage_status(benchmark, stage_name, position, len(stage_plan), results,
                               is_root, archive["check5_gate"])

        if status:
            benchmark.set_node_status(archive, node_id, status)
            benchmark.save_archive(archive, paths.archive_path)
            return status, total_in, total_out

        benchmark.save_archive(archive, paths.archive_path)
        prior_merged = merged_path

    return "alive", total_in, total_out


def _bootstrap(benchmark, archive: dict, paths, max_workers: int, steps_bonus_mode: str):
    genome_path = benchmark.build_root_genome(
        paths.project_root, os.path.join(paths.cycle_dir("root"), "genome.json"))
    root_id = benchmark.add_node(archive, None, genome_path, cycle_index=-1)
    benchmark.get_node(archive, root_id)["mutation"] = {"outcome": "baseline"}
    benchmark.save_archive(archive, paths.archive_path)

    print(f"\n--- BOOTSTRAP: evaluating the baseline agent ({root_id}) ---")
    status, tokens_in, tokens_out = _evaluate_node(
        benchmark, genome_path, root_id, archive, paths, max_workers,
        is_root=True, steps_bonus_mode=steps_bonus_mode)

    root = benchmark.get_node(archive, root_id)
    check5_passed = root["stages"].get(benchmark.FIRST_STAGE_NAME, {}).get("passed", 0)
    archive["check5_gate"] = check5_passed
    benchmark.save_archive(archive, paths.archive_path)

    return root_id, status, tokens_in, tokens_out, check5_passed


def _bootstrap_from_baseline(benchmark, archive: dict, paths, baseline_data: dict,
                             steps_bonus_mode: str):
    """
    Seeds the root node from a saved baseline (see src/eval/baseline_archive.py)
    instead of evaluating it again. No episodes are played.
    """
    genome_path = benchmark.build_root_genome(
        paths.project_root, os.path.join(paths.cycle_dir("root"), "genome.json"))
    root_id = benchmark.add_node(archive, None, genome_path, cycle_index=-1)
    benchmark.get_node(archive, root_id)["mutation"] = {
        "outcome": "baseline (frozen)",
        "baseline_established_at": baseline_data["established_at"],
        "baseline_repeats": baseline_data["repeats"],
    }

    print(f"\n--- BOOTSTRAP: importing frozen baseline for {root_id} "
         f"(established {baseline_data['established_at']}, "
         f"{baseline_data['repeats']} repeat(s)) ---")
    # The baseline only holds the stages its own bootstrap reached, so replay the
    # same gate logic and stop where it stopped.
    stage_plan = archive["game_pool"]["stage_plan"]
    status = "alive"
    for position, (stage_name, _size) in enumerate(stage_plan):
        stage = baseline_data["stages"].get(stage_name)
        if stage is None:
            break
        benchmark.record_stage_result(
            archive, root_id, stage_name, stage,
            stage.get("predictions_path"), stage.get("eval_log_path"),
            stage.get("results_json_path"), steps_bonus_mode)
        stage_status = _stage_status(benchmark, stage_name, position, len(stage_plan), stage,
                                     is_root=True, check5_gate=archive.get("check5_gate") or 0)
        if stage_status:
            status = stage_status
            break

    benchmark.set_node_status(archive, root_id, status)
    root = benchmark.get_node(archive, root_id)
    check5_passed = root["stages"].get(benchmark.FIRST_STAGE_NAME, {}).get("passed", 0)
    archive["check5_gate"] = check5_passed
    benchmark.save_archive(archive, paths.archive_path)

    return root_id, status, 0, 0, check5_passed


def establish_baseline_for_benchmark(paths, benchmark_name: str, max_workers: int,
                                     steps_bonus_mode: str, repeats: int) -> str:
    """
    Runs `repeats` independent bootstrap evaluations of the unmutated agent and
    saves their average as the baseline for this benchmark and model pair. Each
    repeat writes to its own subdirectory of the run.
    """
    from src.config.llm_config import describe_active_models
    from src.eval.baseline_archive import aggregate_stage_results, save_baseline
    from src.utils.run_context import RunPaths

    benchmark = get_benchmark(benchmark_name)
    models = describe_active_models()

    per_stage_repeats = {}
    for i in range(repeats):
        repeat_dir = os.path.join(paths.run_dir, f"repeat_{i}")
        repeat_paths = RunPaths(
            project_root=paths.project_root,
            run_dir=repeat_dir,
            logs_dir=os.path.join(repeat_dir, "logs"),
            episodes_dir=os.path.join(repeat_dir, "logs", "episodes"),
            results_dir=os.path.join(repeat_dir, "eval_results"),
            artifacts_dir=os.path.join(repeat_dir, "artifacts"),
            archive_path=os.path.join(repeat_dir, "archive.json"),
            fallback_log=paths.fallback_log,
        )
        archive = benchmark.init_archive(split=benchmark.VALIDATION_SPLIT,
                                         archive_path=repeat_paths.archive_path)
        print(f"\n=== Establishing baseline: repeat {i + 1}/{repeats} ===")
        root_id, _status, _tokens_in, _tokens_out, _check5 = _bootstrap(
            benchmark, archive, repeat_paths, max_workers, steps_bonus_mode)
        root = benchmark.get_node(archive, root_id)
        for stage_name, stage in root["stages"].items():
            per_stage_repeats.setdefault(stage_name, []).append(stage)

    aggregated_stages = {}
    for stage_name, repeat_stages in per_stage_repeats.items():
        aggregated = aggregate_stage_results(stage_name, repeat_stages)
        # File pointers of the last repeat, kept for inspection only.
        last = repeat_stages[-1]
        aggregated["predictions_path"] = last.get("predictions_path")
        aggregated["eval_log_path"] = last.get("eval_log_path")
        aggregated["results_json_path"] = last.get("results_json_path")
        aggregated_stages[stage_name] = aggregated

    saved_path = save_baseline(paths.project_root, benchmark_name, models,
                               aggregated_stages, repeats, per_stage_repeats)
    print(f"\nBaseline established for {benchmark_name} under {models}: {saved_path}")
    for stage_name, stage in aggregated_stages.items():
        print(f"  {stage_name}: {stage['passed']}/{stage['total']} "
             f"({stage['pass_rate']:.1%}) averaged over {repeats} repeat(s)")
    return saved_path


def run_benchmark_evolution(paths, benchmark_name: str = "alfworld", max_workers: int = 8,
                            max_cycles: int = None, max_tokens: int = None,
                            steps_bonus_mode: str = "raw", mining_source: str = "train",
                            meta_mode: str = "recursive", use_baseline: bool = False,
                            parent_selection: str = "weighted"):
    benchmark = get_benchmark(benchmark_name)
    total_tokens = 0
    cost_log = []
    consecutive_failures = 0
    stop_reason = "all requested cycles completed"

    if os.path.exists(paths.archive_path):
        archive = benchmark.load_archive(paths.archive_path)
        benchmark.assert_archive_matches_active_models(archive, paths.archive_path)
        total_tokens = archive.get("total_tokens", 0)
        print(f"Resumed archive: {len(archive['nodes'])} nodes, models {archive['models']}, "
             f"{total_tokens:,} tokens already spent")
    else:
        archive = benchmark.init_archive(split=benchmark.VALIDATION_SPLIT, archive_path=paths.archive_path)
        pool = archive["game_pool"]
        print(f"Game pool: {pool['total_pool_size']} games {pool['per_type_counts']}")
        print(f"Stage plan: {pool['stage_plan']}")

        if use_baseline:
            from src.eval.baseline_archive import load_baseline

            baseline_data = load_baseline(paths.project_root, benchmark_name, archive["models"])
            if baseline_data is None:
                raise RuntimeError(
                    f"--use_baseline was requested but no frozen baseline exists yet for "
                    f"{benchmark_name} under {archive['models']}. Run with "
                    f"--establish_baseline first (see run_evolution.py --help).")
            if baseline_data["models"] != archive["models"]:
                raise RuntimeError(
                    f"Frozen baseline model mismatch for {benchmark_name}.\n"
                    f"  baseline was established with: {baseline_data['models']}\n"
                    f"  current configuration        : {archive['models']}\n"
                    f"Re-run --establish_baseline for this configuration.")
            root_id, status, tokens_in, tokens_out, check5_passed = _bootstrap_from_baseline(
                benchmark, archive, paths, baseline_data, steps_bonus_mode)
        else:
            root_id, status, tokens_in, tokens_out, check5_passed = _bootstrap(
                benchmark, archive, paths, max_workers, steps_bonus_mode)
        total_tokens += tokens_in + tokens_out
        archive["total_tokens"] = total_tokens
        cost_log.append({"label": f"bootstrap ({root_id}, {status})",
                         "meta_tokens_in": 0, "meta_tokens_out": 0,
                         "episode_tokens_in": tokens_in, "episode_tokens_out": tokens_out,
                         "total": tokens_in + tokens_out})
        if check5_passed == 0:
            print(f"The baseline agent solved 0/{archive['game_pool']['stage_plan'][0][1]} of the "
                  f"{benchmark.FIRST_STAGE_NAME} games, so there is no lineage to evolve from. "
                  f"Aborting.")
            stop_reason = f"baseline failed the {benchmark.FIRST_STAGE_NAME} gate"
            _finish(benchmark, archive, paths, cost_log, total_tokens, stop_reason,
                   cycles=0, max_workers=max_workers)
            return

    print(f"Mining source: {mining_source} | Meta-agent mode: {meta_mode}")
    train_pool = benchmark.build_train_pool() if mining_source == "train" else []
    if mining_source == "train":
        print(f"Train pool for mining: {len(train_pool)} games")

    cycle = 0
    while True:
        if max_cycles is not None and cycle >= max_cycles:
            stop_reason = f"reached the {max_cycles}-cycle limit"
            break
        if max_tokens is not None and total_tokens >= max_tokens:
            stop_reason = f"reached the {max_tokens:,}-token budget"
            break
        cycle += 1

        print(f"\n--- CYCLE {cycle}"
              f"{f'/{max_cycles}' if max_cycles else ''} "
              f"(tokens used {total_tokens:,}"
              f"{f'/{max_tokens:,}' if max_tokens else ''}) ---")

        parent_id = benchmark.select_parent(archive, steps_bonus_mode=steps_bonus_mode,
                                            parent_selection=parent_selection)
        parent = benchmark.get_node(archive, parent_id)
        artifacts_dir = paths.cycle_dir(f"cycle_{cycle:03d}")

        (meta_tokens_in, meta_tokens_out, fallbacks, child_path, mutation) = _run_meta_agents(
            benchmark, parent, train_pool, artifacts_dir, cycle, paths, mining_source, meta_mode,
            max_tokens, total_tokens)
        meta_tokens = meta_tokens_in + meta_tokens_out
        total_tokens += meta_tokens
        archive["total_tokens"] = total_tokens
        over_budget = max_tokens is not None and total_tokens >= max_tokens

        if mutation["outcome"] == "train_pool_exhausted":
            benchmark.set_node_status(archive, parent_id, "train_pool_exhausted")
            benchmark.save_archive(archive, paths.archive_path)

        if child_path is None:
            consecutive_failures += 1
            cost_log.append({"label": f"cycle {cycle} ({mutation['outcome']})",
                             "meta_tokens_in": meta_tokens_in, "meta_tokens_out": meta_tokens_out,
                             "episode_tokens_in": 0, "episode_tokens_out": 0,
                             "total": meta_tokens})
            print(f"  no child produced ({mutation['outcome']}) "
                  f"({consecutive_failures} consecutive so far)")
            if over_budget:
                stop_reason = f"reached the {max_tokens:,}-token budget"
                benchmark.save_archive(archive, paths.archive_path)
                break
            continue

        if over_budget:
            # The meta-agents already exhausted the budget, so the child is not evaluated.
            cost_log.append({"label": f"cycle {cycle} (mutation discarded, budget reached)",
                             "meta_tokens_in": meta_tokens_in, "meta_tokens_out": meta_tokens_out,
                             "episode_tokens_in": 0, "episode_tokens_out": 0,
                             "total": meta_tokens})
            print(f"  cycle {cycle} produced a mutation but the {max_tokens:,}-token budget is "
                  f"already spent ({total_tokens:,} used), discarding it unevaluated")
            stop_reason = f"reached the {max_tokens:,}-token budget"
            benchmark.save_archive(archive, paths.archive_path)
            break

        consecutive_failures = 0
        child_id = benchmark.add_node(archive, parent_id, child_path, cycle_index=cycle - 1)
        child = benchmark.get_node(archive, child_id)
        child["no_think_fallbacks"] = fallbacks
        child["mutation"] = mutation
        benchmark.save_archive(archive, paths.archive_path)

        try:
            status, tokens_in, tokens_out = _evaluate_node(
                benchmark, child_path, child_id, archive, paths, max_workers,
                is_root=False, steps_bonus_mode=steps_bonus_mode)
        except RuntimeError as e:
            # Every episode of a stage crashed, i.e. the mutated code is broken.
            # Discard the child and keep evolving.
            print(f"  {child_id}: all episodes crashed, discarding ({e})")
            benchmark.set_node_status(archive, child_id, "discarded_crash")
            benchmark.save_archive(archive, paths.archive_path)
            cost_log.append({"label": f"cycle {cycle} ({child_id}, discarded_crash)",
                             "meta_tokens_in": meta_tokens_in, "meta_tokens_out": meta_tokens_out,
                             "episode_tokens_in": 0, "episode_tokens_out": 0,
                             "total": meta_tokens})
            if over_budget:
                stop_reason = f"reached the {max_tokens:,}-token budget"
                break
            continue
        episode_tokens = tokens_in + tokens_out
        total_tokens += episode_tokens
        archive["total_tokens"] = total_tokens

        if status != f"discarded_{benchmark.FIRST_STAGE_NAME}":
            benchmark.increment_children_count(archive, parent_id, steps_bonus_mode)
        benchmark.save_archive(archive, paths.archive_path)

        cost_log.append({"label": f"cycle {cycle} ({child_id}, {status})",
                         "meta_tokens_in": meta_tokens_in, "meta_tokens_out": meta_tokens_out,
                         "episode_tokens_in": tokens_in, "episode_tokens_out": tokens_out,
                         "total": meta_tokens + episode_tokens})
        print(f"  {child_id}: {status}, score {child['score']:.1%}"
              f" (parent {parent_id} scored {parent['score']:.1%})")

    _finish(benchmark, archive, paths, cost_log, total_tokens, stop_reason,
           cycles=cycle, max_workers=max_workers)


def _run_meta_agents(benchmark, parent: dict, train_pool: list, artifacts_dir: str,
                     cycle: int, paths, mining_source: str, meta_mode: str,
                     max_tokens: int = None, tokens_before_cycle: int = 0):
    """
    Mining -> Problem Understanding -> Solution Proposer -> Implementation.

    Any failure in these steps (meta-agent error, exhausted training pool,
    context overflow in direct mode, nothing to mine) ends only this cycle and
    is recorded in the returned mutation record.

    max_tokens and tokens_before_cycle bound each recursive meta-agent's token
    spend to the remaining budget plus MAX_CYCLE_OVERAGE_TOKENS.

    Returns (tokens_in, tokens_out, no_think_fallbacks, child_genome_path or
    None, mutation record).
    """
    child_path = os.path.join(artifacts_dir, "genome.json")
    mining_tokens_in = mining_tokens_out = 0
    fallbacks = 0
    problem = solution = ""
    agents = []

    def spent():
        return (mining_tokens_in + sum(a.input_tokens for a in agents),
                mining_tokens_out + sum(a.output_tokens for a in agents))

    def remaining_budget():
        if max_tokens is None:
            return None
        spent_in, spent_out = spent()
        return max(0, max_tokens + MAX_CYCLE_OVERAGE_TOKENS
                     - tokens_before_cycle - spent_in - spent_out)

    try:
        tag = f"{parent['id']}_c{cycle}_mine"
        if mining_source == "validation":
            mining = benchmark.mine_error_samples_from_validation(parent, benchmark.MINING_N,
                                                                   paths.results_dir, tag)
        else:
            mining = benchmark.mine_error_samples(parent["agent_json_path"], train_pool, "train",
                                                  benchmark.MINING_N, paths.results_dir,
                                                  paths.episodes_dir, tag)
        mining_tokens_in, mining_tokens_out = mining["tokens_in"], mining["tokens_out"]

        if mining.get("n_found", benchmark.MINING_N) == 0:
            print(f"  no {mining_source} failures to mine for {parent['id']}, skipping this cycle")
            return (mining_tokens_in, mining_tokens_out, 0, None,
                   {"outcome": "no_validation_failures_to_mine"})

        if mining_source == "validation":
            eval_log_path = mining["eval_log_path"]
        else:
            eval_log_path, _, _ = benchmark.evaluate_predictions(
                mining["samples_path"], f"{parent['id']}_c{cycle}_mine",
                paths.logs_dir, paths.results_dir)

        briefing = benchmark.build_mining_briefing(parent, mining["attempted"], source=mining_source)

        understanding = ProblemUnderstandingAgent(
            agent_json_path=parent["agent_json_path"],
            eval_log_path=eval_log_path,
            predictions_path=mining["samples_path"],
            artifacts_dir=artifacts_dir, mode=meta_mode, token_budget=remaining_budget())
        agents.append(understanding)
        report_path, tokens_in, tokens_out = understanding.run(briefing)
        fallbacks += understanding.no_think_fallbacks
        problem = summarize_problem(understanding.final_text)

        proposer = SolutionProposerAgent(
            agent_json_path=parent["agent_json_path"],
            understanding_report_path=report_path,
            artifacts_dir=artifacts_dir, benchmark=benchmark, mode=meta_mode,
            token_budget=remaining_budget())
        agents.append(proposer)
        plan_path, tokens_in, tokens_out = proposer.run()
        fallbacks += proposer.no_think_fallbacks
        solution = summarize_solution(proposer.parsed_plan(), proposer.final_text)

        implementation = ImplementationAgent(
            plan_path=plan_path,
            parent_json_path=parent["agent_json_path"],
            child_json_path=child_path,
            artifacts_dir=artifacts_dir, mode=meta_mode, token_budget=remaining_budget())
        agents.append(implementation)
        result_path, tokens_in, tokens_out = implementation.run()
        fallbacks += implementation.no_think_fallbacks

    except TrainPoolExhaustedError as e:
        tokens_in, tokens_out = spent()
        fallbacks = sum(a.no_think_fallbacks for a in agents)
        print_meta_summary(cycle, parent["id"], problem, solution, fallbacks, tokens_in + tokens_out)
        print(f"  TRAIN POOL EXHAUSTED for {parent['id']}: {e}")
        return (tokens_in, tokens_out, fallbacks, None,
               {"outcome": "train_pool_exhausted", "error": str(e)[:500]})

    except ContextLengthError as e:
        tokens_in, tokens_out = spent()
        fallbacks = sum(a.no_think_fallbacks for a in agents)
        print_meta_summary(cycle, parent["id"], problem, solution, fallbacks, tokens_in + tokens_out)
        print(f"  META CONTEXT LENGTH ERROR: {e}")
        return (tokens_in, tokens_out, fallbacks, None,
               {"outcome": "context_length_error", "error": str(e)[:500],
                "problem": problem, "solution": solution})

    except Exception as e:  # noqa: BLE001 - any meta-agent failure ends only this cycle
        tokens_in, tokens_out = spent()
        fallbacks = sum(a.no_think_fallbacks for a in agents)
        print_meta_summary(cycle, parent["id"], problem, solution, fallbacks, tokens_in + tokens_out)
        print(f"  META-AGENT FAILURE: {type(e).__name__}: {e}")
        return (tokens_in, tokens_out, fallbacks, None,
               {"outcome": "meta-agent failure", "error": f"{type(e).__name__}: {str(e)[:500]}",
                "problem": problem, "solution": solution})

    tokens_in, tokens_out = spent()
    print_meta_summary(cycle, parent["id"], problem, solution, fallbacks, tokens_in + tokens_out)
    if result_path is None:
        print("  the implementation agent changed nothing, no child to evaluate")
        return (tokens_in, tokens_out, fallbacks, None,
               {"outcome": "no change applied", "problem": problem, "solution": solution})

    return (tokens_in, tokens_out, fallbacks, result_path,
           {"outcome": "mutation applied", "problem": problem, "solution": solution})


def run_test_set_comparison(benchmark, archive: dict, paths, root: dict, best: dict,
                             max_workers: int):
    """
    Plays the root and the best node on the held-out test split and reports the
    comparison.
    """
    test_dir = os.path.join(paths.run_dir, "test_results")
    test_logs = os.path.join(test_dir, "logs")
    test_episodes = os.path.join(test_dir, "logs", "episodes")
    os.makedirs(test_episodes, exist_ok=True)

    test_games = benchmark.test_task_pool()
    print(f"\n--- TEST SET: {len(test_games)} held-out tasks ---")

    def _play(genome_path: str, tag: str):
        predictions_path, _, _ = benchmark.run_episodes(
            agent_json_path=genome_path, game_files=test_games, split=benchmark.TEST_SPLIT,
            max_workers=max_workers, results_dir=test_dir, episodes_dir=test_episodes, tag=tag)
        _, results_json_path, results = benchmark.evaluate_predictions(
            predictions_path, tag, test_logs, test_dir)
        return results_json_path, results

    root_path, root_results = _play(root["agent_json_path"], "root")
    print(f"  root {root['id']}: {root_results['pass_rate']:.1%} on the test set")

    best_path, best_results = _play(best["agent_json_path"], best["id"])
    print(f"  best {best['id']}: {best_results['pass_rate']:.1%} on the test set")
    print(benchmark.compare_results(root_path, best_path))


def _finish(benchmark, archive: dict, paths, cost_log: list,
           total_tokens: int, stop_reason: str, cycles: int, max_workers: int):
    write_final_report(benchmark, archive, paths, read_fallbacks(paths.fallback_log),
                       cost_log, total_tokens, stop_reason, cycles)

    scored = [n for n in archive["nodes"] if n["score"] is not None]
    root = archive["nodes"][0]
    if not scored:
        return
    best = benchmark.pick_best_node(scored)

    if best["id"] != root["id"]:
        shared = [s for s in reversed(benchmark.stage_plan_names(archive))
                 if s in root["stages"] and s in best["stages"]]
        if shared:
            print(f"\nBaseline vs best on {shared[0]}:")
            print(benchmark.compare_results(root["stages"][shared[0]]["results_json_path"],
                                            best["stages"][shared[0]]["results_json_path"]))

        run_test_set_comparison(benchmark, archive, paths, root, best, max_workers)
