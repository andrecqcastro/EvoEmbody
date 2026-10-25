"""
Paired bootstrap confidence intervals for a finished run's root-vs-best
held-out test comparison. Post-processing only: no model calls.

    python scripts/bootstrap_test_ci.py --run_tag qwen_alfworld --n_resamples 5000
    python scripts/bootstrap_test_ci.py --run_dir runs/<run_tag>/run_<timestamp>/

Reads test_results/results_root.json and the best node's results_<id>.json.
Both agents were played on the same test tasks, so outcomes are paired by
instance_id and each resample draws the same task indices for both.

Saves the result to <run_dir>/test_results/confidence_intervals.json.
"""
import argparse
import json
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.run_context import latest_run_dir


def _load_episodes(results_path: str) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["episodes"]


def _find_test_results(run_dir: str, best_id: str = None):
    """Returns (root_path, best_path). Without best_id, the only non-root results file is used."""
    test_dir = os.path.join(run_dir, "test_results")
    root_path = os.path.join(test_dir, "results_root.json")
    if not os.path.exists(root_path):
        raise SystemExit(f"No {root_path}: this run has no held-out test-set comparison yet "
                         f"(it needs to have finished with more than one scored node).")

    if best_id:
        best_path = os.path.join(test_dir, f"results_{best_id}.json")
        if not os.path.exists(best_path):
            raise SystemExit(f"No {best_path} for --best_id {best_id!r}.")
        return root_path, best_path

    others = [f for f in os.listdir(test_dir)
             if f.startswith("results_") and f.endswith(".json") and f != "results_root.json"]
    if len(others) != 1:
        raise SystemExit(
            f"Expected exactly one non-root results_*.json in {test_dir}, found {others}. "
            f"Pass --best_id explicitly (the node id in its filename, e.g. "
            f"results_alfworld_v27.json -> --best_id alfworld_v27).")
    return root_path, os.path.join(test_dir, others[0])


def paired_outcomes(root_episodes: dict, best_episodes: dict):
    """Aligns both runs by instance_id and returns two 0/1 outcome lists in the same order."""
    root_ids, best_ids = set(root_episodes), set(best_episodes)
    if root_ids != best_ids:
        only_root = root_ids - best_ids
        only_best = best_ids - root_ids
        raise SystemExit(
            f"root and best were not scored on the same task set, cannot pair them. "
            f"Only in root: {sorted(only_root)[:5]}... Only in best: {sorted(only_best)[:5]}...")

    ordered_ids = sorted(root_ids)
    root_outcomes = [int(bool(root_episodes[i]["success"])) for i in ordered_ids]
    best_outcomes = [int(bool(best_episodes[i]["success"])) for i in ordered_ids]
    return root_outcomes, best_outcomes


def bootstrap_ci(root_outcomes: list, best_outcomes: list, n_resamples: int, ci: float, seed: int):
    """
    Paired bootstrap over task indices. Returns the point estimate and percentile
    CI for root, best, and their difference.
    """
    rng = random.Random(seed)
    n = len(root_outcomes)
    alpha = (1 - ci) / 2

    root_rates, best_rates, diffs = [], [], []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        r_rate = sum(root_outcomes[i] for i in idx) / n
        b_rate = sum(best_outcomes[i] for i in idx) / n
        root_rates.append(r_rate)
        best_rates.append(b_rate)
        diffs.append(b_rate - r_rate)

    def percentile_ci(values):
        values = sorted(values)
        lo = values[int(alpha * n_resamples)]
        hi = values[int((1 - alpha) * n_resamples) - 1]
        return lo, hi

    return {
        "n_tasks": n,
        "n_resamples": n_resamples,
        "ci": ci,
        "root": {"point": sum(root_outcomes) / n, "ci": percentile_ci(root_rates)},
        "best": {"point": sum(best_outcomes) / n, "ci": percentile_ci(best_rates)},
        "diff": {"point": (sum(best_outcomes) - sum(root_outcomes)) / n, "ci": percentile_ci(diffs)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_tag", default=None,
                        help="Finds the most recent finished run under this tag "
                             "(runs/<run_tag>/run_<timestamp>/). Mutually exclusive with --run_dir.")
    parser.add_argument("--run_dir", default=None,
                        help="A specific run directory, instead of --run_tag + auto-locate.")
    parser.add_argument("--best_id", default=None,
                        help="The best node's id (e.g. alfworld_v27), when test_results/ holds "
                             "more than one non-root results file.")
    parser.add_argument("--n_resamples", type=int, default=5000,
                        help="Bootstrap resamples. Default: 5000.")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="Confidence level, e.g. 0.95 for a 95%% interval. Default: 0.95.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Resampling RNG seed, for a reproducible report. Default: 0.")
    args = parser.parse_args()

    if bool(args.run_tag) == bool(args.run_dir):
        parser.error("Pass exactly one of --run_tag or --run_dir.")

    if args.run_tag:
        run_dir = latest_run_dir(PROJECT_ROOT, args.run_tag)
        if run_dir is None:
            raise SystemExit(f"No finished run with an archive found under runs/{args.run_tag}/.")
    else:
        run_dir = args.run_dir

    root_path, best_path = _find_test_results(run_dir, args.best_id)
    print(f"root: {root_path}\nbest: {best_path}")

    root_episodes = _load_episodes(root_path)
    best_episodes = _load_episodes(best_path)
    root_outcomes, best_outcomes = paired_outcomes(root_episodes, best_episodes)

    result = bootstrap_ci(root_outcomes, best_outcomes, args.n_resamples, args.ci, args.seed)

    out_path = os.path.join(run_dir, "test_results", "confidence_intervals.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    pct = result["ci"] * 100
    print(f"\n{result['n_tasks']} held-out tasks, {result['n_resamples']} paired bootstrap resamples "
         f"(seed={args.seed})\n")
    for label in ("root", "best"):
        point = result[label]["point"]
        lo, hi = result[label]["ci"]
        print(f"  {label:5s}: {point:.1%}  [{pct:.0f}% CI {lo:.1%} - {hi:.1%}]")
    d_point = result["diff"]["point"]
    d_lo, d_hi = result["diff"]["ci"]
    crosses_zero = d_lo <= 0 <= d_hi
    print(f"  diff  (best - root): {d_point:+.1%}  [{pct:.0f}% CI {d_lo:+.1%} - {d_hi:+.1%}]"
         f"{'  (CI includes 0: not distinguishable from no change)' if crosses_zero else ''}")


if __name__ == "__main__":
    main()
