"""
Problem Understanding: inspects the mined failures and the agent's source and
writes a diagnostic report for the Solution Proposer.
"""
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.meta_loop import DirectAnswer, MetaLoop
from src.tools.eval_inspector import evaluation_tools
from src.tools.repo_inspector import inspection_tools
from src.utils.agent_json import read_genome
from src.utils.llm_response_guard import (
    MIN_USEFUL_OUTPUT_CHARS, empty_output_error, response_text,
)

SYSTEM_PROMPT = """
You are a Problem Understanding Agent. An agent has been run on a handful of
tasks in a text-based interactive environment, and every
one of them failed. Your only job is to find out why and name the real root
cause, from those episodes and the agent's own source code. You diagnose
only — proposing or applying a fix is someone else's job.

SCOPE: you can inspect the agent's own source code and its evaluation output.
There is no other repository.

TOOLS:
- Evaluation: list_prediction_ids, get_prediction, read_episode_log, read_log_tail, search_log
- Code: list_repository_structure, get_file_outline, get_entity_info, get_method_body,
  get_full_file, get_imports, get_call_graph

METHOD:
1. list_prediction_ids to see the failed episodes you were given.
2. get_prediction and read_episode_log on each of them. Read what the model
   actually wrote and what the environment answered.
3. Read the agent's source and its prompt/parsing code.
4. Name the single root cause shared by the largest group of failures. That is
   the deliverable — find it wherever it actually is.

A few angles that often help point toward the real cause, so use whichever are
relevant to what you see in the episodes:
- could the agent reach the goal in fewer steps?
- is the environment only partially observable, and does the agent's approach
  account for what it cannot yet see?
- does this depend on remembering something from earlier turns or episodes,
  and does the agent do that well?
- does the agent express or act on uncertainty about its own choices well?

FINAL ANSWER — plain text, in this shape:
ROOT CAUSE: one paragraph naming the structural property of the code that causes
the failures. Not an episode detail.
DIRECT ERRORS: what demonstrably went wrong in this evaluation.
POTENTIAL ERRORS: what the code would also get wrong in other situations.

You have a 100-interaction limit. Answer before you reach it.
"""

DIRECT_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "TOOLS:\n- Evaluation: list_prediction_ids, get_prediction, read_episode_log, read_log_tail, "
    "search_log\n- Code: list_repository_structure, get_file_outline, get_entity_info, "
    "get_method_body,\n  get_full_file, get_imports, get_call_graph",
    "You are given the full failed-episode evidence and the full agent source directly "
    "in this message — there are no tools and no lookups. Read all of it before answering.",
).replace(
    "METHOD:\n1. list_prediction_ids to see the failed episodes you were given.\n"
    "2. get_prediction and read_episode_log on each of them. Read what the model\n"
    "   actually wrote and what the environment answered.\n"
    "3. Read the agent's source and its prompt/parsing code.\n"
    "4. Name the single root cause shared by the largest group of failures. That is\n"
    "   the deliverable — find it wherever it actually is.",
    "METHOD:\n1. Read every failed episode record and its full log below.\n"
    "2. Read the agent's source and its prompt/parsing code, also below.\n"
    "3. Name the single root cause shared by the largest group of failures. That is\n"
    "   the deliverable — find it wherever it actually is.",
).replace("You have a 100-interaction limit. Answer before you reach it.\n", "")


class ProblemUnderstandingAgent:
    def __init__(self, agent_json_path: str, eval_log_path: str, predictions_path: str,
                 artifacts_dir: str, mode: str = "recursive", token_budget: int = None):
        self.agent_json_path = agent_json_path
        self.eval_log_path = eval_log_path
        self.predictions_path = predictions_path
        self.mode = mode
        self.token_budget = token_budget
        self.report_path = os.path.join(artifacts_dir, "understanding_report.txt")
        self.trace_path = os.path.join(artifacts_dir, "understanding_trace.txt")
        self.no_think_fallbacks = 0
        self.final_text = ""
        self.loop = None
        self.input_tokens = 0
        self.output_tokens = 0

    def _inline_evidence(self) -> str:
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        blocks = []
        for record in records:
            log_path = record.get("episode_log")
            log_text = ""
            if log_path and os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    log_text = f.read()
            blocks.append(
                f"--- {record.get('instance_id')} ({record.get('task_type')}) ---\n"
                f"Instruction: {record.get('task_instruction')}\n"
                f"Steps: {record.get('num_steps')}\n\n"
                f"Episode log:\n{log_text}")
        return "FAILED EPISODES:\n\n" + "\n\n".join(blocks)

    @staticmethod
    def _inline_source(genome: dict) -> str:
        parts = [f"=== {path} ===\n{entry['content']}"
                for path, entry in genome.get("files", {}).items()]
        return "AGENT SOURCE:\n\n" + "\n\n".join(parts)

    def run(self, task_description: str):
        """Returns (report_path, input_tokens, output_tokens)."""
        genome = read_genome(self.agent_json_path)
        final = None

        if self.mode == "direct":
            driver = DirectAnswer(name="ProblemUnderstandingAgent", temperature=0.1,
                                  trace_path=self.trace_path)
            prompt = "\n\n".join([task_description, self._inline_evidence(),
                                  self._inline_source(genome),
                                  "Give your final answer now, in the exact format specified."])
            messages = [SystemMessage(content=DIRECT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            final = driver.run(messages)
            self.no_think_fallbacks = driver.no_think_fallbacks
            self.final_text = response_text(final).strip()
            input_tokens, output_tokens = driver.input_tokens, driver.output_tokens
        else:
            tools = inspection_tools(genome) + evaluation_tools(self.eval_log_path,
                                                                self.predictions_path)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=task_description),
            ]
            self.loop = loop = MetaLoop(name="ProblemUnderstandingAgent", tools=tools,
                                        temperature=0.1, trace_path=self.trace_path,
                                        token_budget=self.token_budget)
            final = loop.run(messages)
            self.no_think_fallbacks = loop.no_think_fallbacks
            self.final_text = response_text(final).strip()
            input_tokens, output_tokens = loop.input_tokens, loop.output_tokens

        if len(self.final_text) < MIN_USEFUL_OUTPUT_CHARS:
            raise empty_output_error("ProblemUnderstandingAgent", final)

        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(self.final_text + "\n")
        return self.report_path, input_tokens, output_tokens
