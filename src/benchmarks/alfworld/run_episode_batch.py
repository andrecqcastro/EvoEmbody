"""
Plays one batch of ALFWorld episodes inside the genome sandbox (GENOME_SANDBOX=on).

    python -m src.benchmarks.alfworld.run_episode_batch --agent-json-path ... --split ...
"""
import argparse
import json
import sys

from src.benchmarks.alfworld.generate_alfworld_predictions import _run_alfworld_episodes_direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-json-path", required=True)
    parser.add_argument("--game-files", required=True, help="JSON list of game file paths")
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    predictions_path, tokens_in, tokens_out = _run_alfworld_episodes_direct(
        agent_json_path=args.agent_json_path,
        game_files=json.loads(args.game_files),
        split=args.split,
        max_workers=args.max_workers,
        results_dir=args.results_dir,
        episodes_dir=args.episodes_dir,
        tag=args.tag,
    )
    print(json.dumps({"predictions_path": predictions_path,
                      "tokens_in": tokens_in, "tokens_out": tokens_out}))


if __name__ == "__main__":
    main()
    sys.exit(0)
