"""
The half of the agent genome that lives outside the agent class: how an episode
turn is presented to the model, and how its reply becomes an environment action.

MUTABLE. The meta-agents may rewrite this file, and a mutated copy is written
back here before the child is evaluated. Keep it self-contained: it is executed
from the mutated JSON, so it must not depend on anything the evaluator owns.
"""

REASONING_PREFIXES = ("thought:", "reasoning:", "reflection:", "observation:", "plan:")

_STRIPPABLE_PREFIXES = (
    "action:", "i will ", "i'll ", "let me ", "next action:",
    "i choose to ", "my action:", "command:", "i should ", "i need to ",
)


def build_alfworld_prompt(task_instruction: str, history: list,
                          admissible_commands: list = None) -> str:
    """
    Renders one turn: the task and the episode so far as a ReAct-style
    Thought / Action / Observation trace, then the question.

    admissible_commands is the exact set of commands the engine will accept this
    turn. The runner always supplies it; this baseline deliberately does not show
    it to the model, so putting it in the prompt remains available as a mutation.

    Each history entry may carry a "thought" — the reasoning the model wrote
    before its action on that turn (see extract_thought below). Showing it back
    here is what keeps prior Thought/Action/Observation steps actually in
    context, rather than only the action and observation surviving.
    """
    lines = [f"TASK: {task_instruction}", "", "INTERACTION HISTORY:"]

    if not history:
        lines.append("(No actions taken yet.)")
    for entry in history:
        action = entry.get("action")
        obs = entry.get("obs", "")
        if action is None:
            lines.append(f"[Start]\n{obs}")
        else:
            thought = entry.get("thought")
            if thought:
                lines.append(f"[Thought] {thought}")
            lines.append(f"[Turn {entry.get('turn', 0)}] > {action}")
            lines.append(obs)

    lines.extend(["", "What is your next action?"])
    return "\n".join(lines)


def extract_thought(response: str) -> str:
    """
    The reasoning the model wrote before its Action: line — the Thought half of
    ReAct's Thought/Action/Observation cycle — kept so it can be shown back to
    the model on later turns instead of being discarded the moment the action is
    parsed out of the same reply.
    """
    lines = []
    for line in response.strip().split("\n"):
        if line.strip().lower().startswith("action:"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_alfworld_action(response: str) -> str:
    """
    Extracts the single command to send to the environment.

    Chain-of-thought replies ("Thought: ...\nAction: go to shelf 1") are supported:
    a line labelled 'Action:' wins, otherwise reasoning lines are skipped and the
    first remaining line is used. Without this the reasoning text reaches the
    engine and every turn answers "Nothing happens".
    """
    action = None
    for line in response.strip().split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            action = stripped[len("action:"):].strip()
            break

    if action is None:
        candidates = [
            line.strip() for line in response.strip().split("\n")
            if line.strip() and not line.strip().lower().startswith(REASONING_PREFIXES)
        ]
        action = candidates[0] if candidates else ""

    for prefix in _STRIPPABLE_PREFIXES:
        if action.lower().startswith(prefix):
            action = action[len(prefix):].strip()
            break

    return action.split("\n")[0].strip(" \t\"'.")
