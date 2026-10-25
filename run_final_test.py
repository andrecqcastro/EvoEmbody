"""
Re-runs the held-out test-set comparison of a finished run, playing its root
and best genomes on the test split without any further evolution.

    python run_final_test.py --benchmark alfworld --run_tag <run_tag>

Overwrites the run's test_results/. The model servers must be running; use
EVOLUTION_ENTRYPOINT=run_final_test.py with scripts/03_run_evolution.sh.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.benchmarks.registry import get_benchmark
from src.eval.evolution_loop import run_test_set_comparison
from src.utils.run_context import RunPaths, latest_run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=["alfworld"],
                        help="The benchmark the finished run was evolved against.")
    parser.add_argument("--run_tag", default=None,
                        help="The --run_tag that finished run used, to find it under runs/.")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Episodes played in parallel.")
    args = parser.parse_args()

    benchmark = get_benchmark(args.benchmark)
    run_dir = latest_run_dir(PROJECT_ROOT, args.run_tag)
    if run_dir is None:
        location = f"runs/{args.run_tag}/" if args.run_tag else "runs/"
        raise SystemExit(f"No finished run with an archive found under {location}.")
    print(f"Re-running the test-set comparison for {run_dir}")

    paths = RunPaths(
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        logs_dir=os.path.join(run_dir, "logs"),
        episodes_dir=os.path.join(run_dir, "logs", "episodes"),
        results_dir=os.path.join(run_dir, "eval_results"),
        artifacts_dir=os.path.join(run_dir, "artifacts"),
        archive_path=os.path.join(run_dir, "archive.json"),
        fallback_log=os.path.join(run_dir, "fallbacks.jsonl"),
    )
    os.environ["ALFWORLD_FALLBACK_LOG"] = paths.fallback_log

    archive = benchmark.load_archive(paths.archive_path)
    root = archive["nodes"][0]
    scored = [n for n in archive["nodes"] if n["score"] is not None]
    if not scored:
        raise SystemExit(f"{run_dir}'s archive has no scored nodes, nothing to test.")
    best = benchmark.pick_best_node(scored)
    print(f"root: {root['id']} | best: {best['id']}"
         + (" (same node)" if best["id"] == root["id"] else ""))

    run_test_set_comparison(benchmark, archive, paths, root, best, args.max_workers)


if __name__ == "__main__":
    main()
