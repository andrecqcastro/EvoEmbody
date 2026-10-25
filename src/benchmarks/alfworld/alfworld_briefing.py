"""
The briefing handed to the Problem Understanding Agent: what the environment is,
how the agent's evidence was gathered, and what the meta-agents may and may not
change.
"""
import json

from src.eval.lineage_archive import highest_completed_stage
from src.benchmarks.alfworld.genome import AGENT_FILE, POLICY_FILE

ENVIRONMENT = f"""
ENVIRONMENT
- Partially observable. The agent cannot see an object until it has gone to the
  location holding it, so exploration is part of every task.
- "Nothing happens" is the engine's reply when an action's precondition is not
  met. It is a signal to change strategy, not to retry.
- The interaction history grows every turn, so a long episode is also an
  expensive one.
- "You won!" means every task predicate is satisfied.

ACTION VOCABULARY (engine-enforced — no other phrasing is accepted)
  go to <location> | take <object> from <location> | move <object> to <receptacle>
  open <container> | clean <object> with <sink> | heat <object> with <microwave>
  cool <object> with <fridge> | use <object> | examine <object> | look
  The placement verb is "move ... to ...". "put" is not recognised.

WHAT YOU MAY CHANGE
- {AGENT_FILE}: the ArchiveAgent class — its system prompt and generate_response.
- {POLICY_FILE}: build_alfworld_prompt, which decides what the model is shown
  each turn (including its own past episodes, if any — see below), and
  parse_alfworld_action, which turns its reply into a command.

WHAT IS FIXED
- The episode runner, the scoring, the step limits and the task set. Do not
  propose changing them.
- parse_alfworld_action already extracts an 'Action:' line from chain-of-thought
  output and skips Thought:/Reasoning:/Reflection: lines. Reasoning formats work;
  do not propose adding support for them. A reply with no action line at all
  parses to an empty action and wastes the turn.
- build_alfworld_prompt receives the engine's admissible_commands for the turn
  but does not show them to the model. Showing them is an available change.
- The baseline agent is ReAct-style: build_alfworld_prompt renders every prior
  turn THIS episode as Thought/Action/Observation, so the model sees its own
  earlier reasoning, not just past actions and observations. There is no memory
  of OTHER episodes — each episode starts with no prior context.
- Scoring is pass rate AND step efficiency, so a fix that solves tasks in more
  steps than needed still costs the agent fitness.
""".strip()


def build_mining_briefing(parent: dict, attempted: int, source: str = "train") -> str:
    """
    The briefing handed to Problem Understanding.

    source="train": failures freshly mined from tasks the agent is never scored on.
    source="validation": the agent's own validation failures.
    """
    if parent.get("score") is not None:
        stage = highest_completed_stage(parent) or "unscored"
        score_line = f"The agent currently scores {parent['score']:.1%} on its validation tier ({stage})."
    else:
        score_line = "The agent has not yet been scored on validation."

    if source == "validation":
        evidence_line = (f"To find concrete failures, {attempted} of the agent's own validation "
                         f"episodes were inspected — episodes it was already scored on.")
    else:
        evidence_line = (f"To find concrete failures, it was run on {attempted} freshly drawn "
                         f"tasks it is never scored on, until 3 of them failed.")

    lines = [
        score_line,
        evidence_line,
        "",
        ENVIRONMENT,
        "",
        "Find the single root cause behind the largest group of failures, and report it in the "
        "format your instructions specify.",
    ]
    return "\n".join(lines)


def compare_results(baseline_path: str, best_path: str) -> str:
    """Episode-level diff between two evaluations of the same game subset."""
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(best_path, "r", encoding="utf-8") as f:
        best = json.load(f)

    improved = regressed = 0
    for episode_id in set(baseline["episodes"]) | set(best["episodes"]):
        was = baseline["episodes"].get(episode_id, {}).get("success", False)
        now = best["episodes"].get(episode_id, {}).get("success", False)
        improved += was is False and now is True
        regressed += was is True and now is False

    net = improved - regressed
    return "\n".join([
        f"  baseline : {baseline['passed']}/{baseline['total']} ({baseline['pass_rate']:.1%}), "
        f"step efficiency {baseline.get('avg_step_efficiency')}",
        f"  best     : {best['passed']}/{best['total']} ({best['pass_rate']:.1%}), "
        f"step efficiency {best.get('avg_step_efficiency')}",
        f"  fixed {improved} episodes, broke {regressed}, net {net:+d}",
    ])
