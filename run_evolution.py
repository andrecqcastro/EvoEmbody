"""
EvoEmbody: self-evolution of an embodied task agent.

    python run_evolution.py --max_cycles 20
    python run_evolution.py --max_tokens 50000000 --max_cycles none
    python run_evolution.py --resume --max_cycles 10
    python run_evolution.py --meta_mode direct --mining_source validation

Each run writes everything it produces under runs/[<run_tag>/]run_<timestamp>/.
Starting from scratch (the default) moves any earlier runs under that same tag
into previous_[<run_tag>_]<timestamp>/.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.eval.evolution_loop import establish_baseline_for_benchmark, run_benchmark_evolution
from src.utils.run_context import start_run


def _limit(value: str):
    """A positive integer, or None for 'no limit' (--max_cycles none)."""
    if value is None or value.strip().lower() in ("none", "off", ""):
        return None
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("a limit must be a positive integer, or 'none'")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max_cycles", type=_limit, default="10",
                        help="Stop after this many evolution cycles. 'none' for no limit. "
                             "Default: 10.")
    parser.add_argument("--max_tokens", type=_limit, default="none",
                        help="Stop once this many tokens have been spent, counted at cycle "
                             "boundaries. 'none' for no limit. Default: none.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the lineage of the most recent run (under this --run_tag, "
                             "if given) instead of starting a new one. Default: start from scratch.")
    parser.add_argument("--run_tag", default=None,
                        help="Puts this run under runs/<run_tag>/ instead of runs/, so separate "
                             "configurations (e.g. an ablation pair) don't archive each other.")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Episodes played in parallel. scripts/03_run_evolution.sh sets "
                             "this from the machine's CPU and memory.")
    parser.add_argument("--steps_bonus_mode", default="raw", choices=["raw", "sigmoid"],
                        help="Transform applied to step efficiency in the parent-selection weight.")
    parser.add_argument("--parent_selection", default="weighted", choices=["weighted", "greedy"],
                        help="How the next parent to mutate is picked. 'weighted' (default): "
                             "weighted-random sample over every eligible node, combining score, "
                             "step efficiency, validation tier and number of children. 'greedy': "
                             "always the highest-scoring eligible node, ties broken by step "
                             "efficiency.")
    parser.add_argument("--benchmark", default="alfworld", choices=["alfworld"],
                        help="Benchmark to evolve against. Default: alfworld.")
    parser.add_argument("--mining_source", default="train", choices=["train", "validation"],
                        help="Where Problem Understanding's failures come from. 'train' (default): "
                             "freshly drawn failures from the training split, disjoint from the "
                             "games a node is scored on. 'validation': the parent's own validation "
                             "failures.")
    parser.add_argument("--meta_mode", default="recursive", choices=["recursive", "direct"],
                        help="'recursive' (default): meta-agents retrieve context on demand through "
                             "tools, with context-overflow recovery. 'direct': all context is "
                             "inlined into a single prompt with no recovery; an overflow is "
                             "recorded as a context_length_error outcome.")
    parser.add_argument("--establish_baseline", action="store_true",
                        help="Instead of evolving, run --baseline_repeats bootstrap evaluations "
                             "of the unmutated agent and save their average under baselines/ as "
                             "the baseline for this benchmark and model pair.")
    parser.add_argument("--baseline_repeats", type=int, default=1,
                        help="Number of bootstrap evaluations averaged by --establish_baseline. "
                             "Default: 1.")
    parser.add_argument("--use_baseline", action="store_true",
                        help="Seed the root node from the saved baseline for this benchmark and "
                             "model pair instead of evaluating it again. Ignored with --resume.")
    args = parser.parse_args()

    if args.establish_baseline and args.resume:
        parser.error("--establish_baseline cannot be combined with --resume.")
    if args.establish_baseline and args.use_baseline:
        parser.error("--establish_baseline and --use_baseline are mutually exclusive.")
    if not args.establish_baseline and args.max_cycles is None and args.max_tokens is None:
        parser.error("--max_cycles and --max_tokens cannot both be 'none'.")

    paths = start_run(PROJECT_ROOT, resume=args.resume, run_tag=args.run_tag)
    print(f"Run directory: {paths.run_dir}")

    if args.establish_baseline:
        establish_baseline_for_benchmark(
            paths=paths,
            benchmark_name=args.benchmark,
            max_workers=args.max_workers,
            steps_bonus_mode=args.steps_bonus_mode,
            repeats=args.baseline_repeats,
        )
        return

    print(f"Limits: max_cycles={args.max_cycles or 'none'}, max_tokens={args.max_tokens or 'none'}")

    run_benchmark_evolution(
        paths=paths,
        benchmark_name=args.benchmark,
        max_workers=args.max_workers,
        max_cycles=args.max_cycles,
        max_tokens=args.max_tokens,
        steps_bonus_mode=args.steps_bonus_mode,
        mining_source=args.mining_source,
        meta_mode=args.meta_mode,
        use_baseline=args.use_baseline,
        parent_selection=args.parent_selection,
    )


if __name__ == "__main__":
    main()
