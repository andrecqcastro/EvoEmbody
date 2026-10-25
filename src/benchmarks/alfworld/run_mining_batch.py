"""
Runs one training-split mining batch inside the genome sandbox (GENOME_SANDBOX=on).

    python -m src.benchmarks.alfworld.run_mining_batch --agent-json-path ... --n 1
"""
import argparse
import json
import sys

from src.benchmarks.alfworld.alfworld_mining import _mine_error_samples_direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-json-path", required=True)
    parser.add_argument("--train-pool", required=True, help="JSON list of game file paths")
    parser.add_argument("--split", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    result = _mine_error_samples_direct(
        agent_json_path=args.agent_json_path,
        train_pool=json.loads(args.train_pool),
        split=args.split,
        n=args.n,
        results_dir=args.results_dir,
        episodes_dir=args.episodes_dir,
        tag=args.tag,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
    sys.exit(0)
