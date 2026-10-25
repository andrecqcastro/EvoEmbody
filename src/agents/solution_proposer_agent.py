"""
Solution Proposer: turns the diagnostic report into a JSON mutation plan of one
or more self-contained changes. It reads the code but not the failure traces.
"""
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.meta_loop import DirectAnswer, MetaLoop
from src.tools.repo_inspector import inspection_tools
from src.utils.agent_json import read_genome
from src.utils.llm_response_guard import (
    MIN_USEFUL_OUTPUT_CHARS, empty_output_error, response_text,
)

# {agent_file}, {policy_file} and {domain_notes} are filled in per benchmark.
_SYSTEM_PROMPT_TEMPLATE = """
You are a Solution Proposer Agent. You are given a report on why an agent
failed at a set of tasks in a partially-observable interactive environment.
Read the agent's code and design the change(s) that fix the root cause, as a
machine-readable mutation plan — you design the fix, you do not apply it.

SCOPE: you can inspect the agent's own source code. There is no other repository.
TOOLS: list_repository_structure, get_file_outline, get_entity_info,
get_method_body, get_full_file, get_imports, get_call_graph.

MODIFIABLE FILES — call list_repository_structure and use the exact paths it returns:
- {agent_file}   the ArchiveAgent class: its system prompt and generate_response
- {policy_file}  the prompt-building function (what the model sees each turn) and
  the action-parsing function (how its reply becomes an environment command)
A new file may be proposed instead, if a change genuinely needs one — but editing
these two is usually the right call.

FINAL ANSWER — a valid JSON object and nothing else:
{{
"root_cause": "why the agent fails, in technical terms",
"mutation_plan": [
    {{
    "file_path": "exact path from list_repository_structure",
    "operation": "MODIFY",
    "description": "one sentence naming the change",
    "instructions": "the logical steps for the implementation agent to follow"
    }}
]
}}

CONSTRAINTS:
- One or more independent, self-contained changes. Each entry in mutation_plan
  is a single conceptual idea — do not bundle unrelated ideas into one entry —
  but you are not limited to a single entry when the report points to more
  than one real cause.
- No code blocks in the plan. Write instructions another model can implement.
- Do not mask failures: no filling missing fields with defaults, no try/except
  that swallows a bad model output and returns a placeholder. A failure must
  reach the evaluator as a failure.
- Do not add retry loops that re-invoke the model inside the agent.
{domain_notes}
You have a 100-interaction limit. Answer before you reach it.
"""


def _build_system_prompt(benchmark, direct: bool) -> str:
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        agent_file=benchmark.AGENT_FILE, policy_file=benchmark.POLICY_FILE,
        domain_notes=benchmark.SOLUTION_PROPOSER_DOMAIN_NOTES)
    if not direct:
        return prompt
    return prompt.replace(
        "TOOLS: list_repository_structure, get_file_outline, get_entity_info,\n"
        "get_method_body, get_full_file, get_imports, get_call_graph.\n\n"
        "MODIFIABLE FILES — call list_repository_structure and use the exact paths it returns:",
        "You are given the full agent source directly in this message — there are no tools and no "
        "lookups.\n\nMODIFIABLE FILES — use these exact paths:",
    ).replace("You have a 100-interaction limit. Answer before you reach it.\n", "")


class SolutionProposerAgent:
    def __init__(self, agent_json_path: str, understanding_report_path: str, artifacts_dir: str,
                 benchmark, mode: str = "recursive", token_budget: int = None):
        self.agent_json_path = agent_json_path
        self.understanding_report_path = understanding_report_path
        self.benchmark = benchmark
        self.mode = mode
        self.token_budget = token_budget
        self.plan_path = os.path.join(artifacts_dir, "mutation_plan.txt")
        self.trace_path = os.path.join(artifacts_dir, "proposer_trace.txt")
        self.no_think_fallbacks = 0
        self.final_text = ""
        self.loop = None
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _inline_source(genome: dict) -> str:
        parts = [f"=== {path} ===\n{entry['content']}"
                for path, entry in genome.get("files", {}).items()]
        return "AGENT SOURCE:\n\n" + "\n\n".join(parts)

    def run(self):
        """Returns (plan_path, input_tokens, output_tokens)."""
        genome = read_genome(self.agent_json_path)
        with open(self.understanding_report_path, "r", encoding="utf-8") as f:
            report = f.read()
        final = None

        if self.mode == "direct":
            driver = DirectAnswer(name="SolutionProposerAgent", temperature=0.1,
                                  trace_path=self.trace_path)
            prompt = "\n\n".join([
                f"Problem understanding report:\n\n{report}",
                self._inline_source(genome),
                "Output the JSON mutation plan now, in the exact format specified.",
            ])
            messages = [SystemMessage(content=_build_system_prompt(self.benchmark, direct=True)),
                       HumanMessage(content=prompt)]
            final = driver.run(messages)
            self.no_think_fallbacks = driver.no_think_fallbacks
            self.final_text = response_text(final).strip()
            input_tokens, output_tokens = driver.input_tokens, driver.output_tokens
        else:
            messages = [
                SystemMessage(content=_build_system_prompt(self.benchmark, direct=False)),
                HumanMessage(content=f"Problem understanding report:\n\n{report}\n\n"
                                     f"Inspect the agent's code and output the JSON mutation plan."),
            ]
            self.loop = loop = MetaLoop(name="SolutionProposerAgent", tools=inspection_tools(genome),
                                        temperature=0.1, trace_path=self.trace_path,
                                        token_budget=self.token_budget)
            final = loop.run(messages)
            self.no_think_fallbacks = loop.no_think_fallbacks
            self.final_text = response_text(final).strip()
            input_tokens, output_tokens = loop.input_tokens, loop.output_tokens

        if len(self.final_text) < MIN_USEFUL_OUTPUT_CHARS:
            raise empty_output_error("SolutionProposerAgent", final)

        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        with open(self.plan_path, "w", encoding="utf-8") as f:
            f.write(self.final_text + "\n")
        return self.plan_path, input_tokens, output_tokens

    def parsed_plan(self) -> dict:
        """The plan as a dict, or {} when the model did not emit clean JSON."""
        text = self.final_text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
