"""
Runs one Implementation Agent inside the genome sandbox (GENOME_SANDBOX=on).

    python -m src.agents.run_implementation_batch --plan-path ... --mode recursive
"""
import argparse
import json
import sys

from src.agents.implementation_agent import ImplementationAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--parent-json-path", required=True)
    parser.add_argument("--child-json-path", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--mode", required=True, choices=["recursive", "direct"])
    parser.add_argument("--token-budget", type=int, default=None)
    args = parser.parse_args()

    agent = ImplementationAgent(plan_path=args.plan_path, parent_json_path=args.parent_json_path,
                                child_json_path=args.child_json_path,
                                artifacts_dir=args.artifacts_dir, mode=args.mode,
                                _sandboxed=True, token_budget=args.token_budget)
    child_path, tokens_in, tokens_out = agent.run()
    print(json.dumps({"child_path": child_path, "tokens_in": tokens_in, "tokens_out": tokens_out,
                      "no_think_fallbacks": agent.no_think_fallbacks}))


if __name__ == "__main__":
    main()
    sys.exit(0)
