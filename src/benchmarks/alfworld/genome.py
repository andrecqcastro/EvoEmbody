"""
ALFWorld's genome: the two source files evolution is allowed to rewrite.

    src/agents/archive/agent.py                 the ArchiveAgent class
    src/benchmarks/alfworld/alfworld_policy.py  prompt building, action parsing
"""
from src.utils import agent_json

AGENT_FILE = "src/agents/archive/agent.py"
POLICY_FILE = "src/benchmarks/alfworld/alfworld_policy.py"
GENOME_FILES = (AGENT_FILE, POLICY_FILE)
REQUIRED_POLICY_FUNCTIONS = ("build_alfworld_prompt", "parse_alfworld_action")

# Benchmark-specific constraints appended to the Solution Proposer's system prompt.
SOLUTION_PROPOSER_DOMAIN_NOTES = """
- generate_response(self, user_input: str) -> str must keep its signature.
- The change must help across ALL task types. Do not hardcode receptacle names,
  object names, or per-task heuristics.
- parse_alfworld_action already handles 'Thought: ... Action: ...' output. Do not
  propose adding that.
- self.llm only supports .invoke(messages) — it is not a LangChain chat model. Do
  not call .bind(), .bind_tools(), or pass it extra keyword arguments.
"""


def build_root_genome(project_root: str, dest_path: str) -> str:
    return agent_json.build_root_genome(project_root, dest_path, GENOME_FILES)


def load_genome(json_path: str):
    return agent_json.load_genome(json_path, AGENT_FILE, POLICY_FILE, REQUIRED_POLICY_FUNCTIONS)
