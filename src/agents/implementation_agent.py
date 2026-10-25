"""
Implementation: applies a mutation plan to the parent genome with read_file and
write_file, and writes the child genome. No child is produced when nothing changed.
"""
import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.agents.meta_loop import DirectAnswer, MetaLoop
from src.tools.repo_editor import RepoEditor
from src.utils.agent_json import read_genome
from src.utils.llm_response_guard import reject_unusable_input, response_text

SYSTEM_PROMPT = """
You are an Implementation Agent. You are given a mutation plan — it may list
more than one change. Apply exactly what it specifies, nothing else.

1. Find the mutation plan in the message.
2. read_file each file any entry names, to see its current content.
3. Make exactly the changes every entry asks for, and nothing else.
4. write_file the updated content back, once per file.

write_file accepts the exact path of an existing file, or a genuinely new path
if the plan calls for one. If read_file or write_file says an existing path was
not found, it lists the valid paths — use one of those rather than guessing
another spelling of an existing file.

Be surgical. Changing anything the plan did not ask for makes the result
impossible to attribute.
"""

DIRECT_SYSTEM_PROMPT = """
You are an Implementation Agent. You are given a mutation plan and the complete
current content of every file you may change — there are no tools and no
lookups. Apply exactly what the plan specifies, nothing else.

Output ONE block per file that changed, in exactly this format, with the file's
COMPLETE new content between the marker and the next one (or the end of your
reply):

### FILE: <exact path from the list you were given>
<complete new content of that file>

Do not include a block for a file you did not change. Be surgical — changing
anything the plan did not ask for makes the result impossible to attribute.
"""

EDITOR_TOOLS = (
    ("read_file", "Read the current content of a file."),
    ("write_file", "Write updated content to a file, existing or (rarely) new."),
)

_FILE_BLOCK_RE = re.compile(r"^### FILE: (.+?)\s*$", re.MULTILINE)


def _parse_file_blocks(text: str) -> dict:
    """Splits a direct-mode '### FILE: <path>' reply into {path: new_content}."""
    matches = list(_FILE_BLOCK_RE.finditer(text))
    blocks = {}
    for i, match in enumerate(matches):
        path = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[path] = text[start:end].strip("\n")
    return blocks


class ImplementationAgent:
    def __init__(self, plan_path: str, parent_json_path: str, child_json_path: str,
                 artifacts_dir: str, mode: str = "recursive", _sandboxed: bool = False,
                 token_budget: int = None):
        self.plan_path = plan_path
        self.parent_json_path = parent_json_path
        self.child_json_path = child_json_path
        self.artifacts_dir = artifacts_dir
        self.mode = mode
        self.token_budget = token_budget
        # True inside the sandbox container, so run() does not dispatch again.
        self._sandboxed = _sandboxed
        self.trace_path = os.path.join(artifacts_dir, "implementation_trace.txt")
        self.no_think_fallbacks = 0
        self.final_text = ""
        self.loop = None
        self.input_tokens = 0
        self.output_tokens = 0

    def run(self):
        """
        Returns (child_json_path or None, input_tokens, output_tokens).

        Runs inside the genome sandbox when it is enabled (see src/utils/sandbox.py).
        """
        if not self._sandboxed:
            from src.utils.sandbox import run_module, sandbox_enabled
            if sandbox_enabled():
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))))
                batch_args = ["--plan-path", self.plan_path,
                             "--parent-json-path", self.parent_json_path,
                             "--child-json-path", self.child_json_path,
                             "--artifacts-dir", self.artifacts_dir, "--mode", self.mode]
                if self.token_budget is not None:
                    batch_args += ["--token-budget", str(self.token_budget)]
                result = run_module(
                    project_root=project_root,
                    module="src.agents.run_implementation_batch",
                    args=batch_args,
                    writable_dirs=[self.artifacts_dir],
                )
                if result["returncode"] != 0:
                    raise RuntimeError(
                        f"sandboxed ImplementationAgent run failed (exit "
                        f"{result['returncode']}):\n{result['stderr'][-4000:]}")
                payload = json.loads(result["stdout"].strip().splitlines()[-1])
                self.no_think_fallbacks = payload["no_think_fallbacks"]
                self.input_tokens, self.output_tokens = payload["tokens_in"], payload["tokens_out"]
                return payload["child_path"], payload["tokens_in"], payload["tokens_out"]

        genome = read_genome(self.parent_json_path)
        with open(self.plan_path, "r", encoding="utf-8") as f:
            plan = f.read()

        reject_unusable_input("ImplementationAgent", self.plan_path, plan)

        # Snapshot before RepoEditor, which edits the genome dict in place.
        before = json.dumps(genome, sort_keys=True, ensure_ascii=False)
        editor = RepoEditor(genome)

        if self.mode == "direct":
            driver = DirectAnswer(name="ImplementationAgent", temperature=0.0,
                                  trace_path=self.trace_path)
            files_block = "\n\n".join(
                f"=== {path} ===\n{entry['content']}" for path, entry in editor.files.items())
            prompt = (f"Mutation plan:\n\n{plan}\n\n"
                     f"Valid file paths: {list(editor.files)}\n\n"
                     f"Current file contents:\n\n{files_block}")
            messages = [SystemMessage(content=DIRECT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            final = driver.run(messages)
            self.no_think_fallbacks = driver.no_think_fallbacks
            self.final_text = response_text(final).strip()
            for path, content in _parse_file_blocks(self.final_text).items():
                editor.write_file(path, content)
            input_tokens, output_tokens = driver.input_tokens, driver.output_tokens
        else:
            tools = [
                StructuredTool.from_function(func=getattr(editor, name), name=name,
                                            description=description)
                for name, description in EDITOR_TOOLS
            ]
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Mutation plan:\n\n{plan}\n\n"
                                     f"Valid file paths: {list(editor.files)}"),
            ]
            # The output is the write_file side effect, so the text-only
            # final-answer fallback is disabled.
            self.loop = loop = MetaLoop(name="ImplementationAgent", tools=tools,
                                        temperature=0.0, trace_path=self.trace_path,
                                        final_answer_fallback=False,
                                        token_budget=self.token_budget)
            final = loop.run(messages)
            self.no_think_fallbacks = loop.no_think_fallbacks
            self.final_text = response_text(final).strip()
            input_tokens, output_tokens = loop.input_tokens, loop.output_tokens

        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        if json.dumps(editor.repo, sort_keys=True, ensure_ascii=False) == before:
            return None, input_tokens, output_tokens

        os.makedirs(os.path.dirname(self.child_json_path), exist_ok=True)
        with open(self.child_json_path, "w", encoding="utf-8") as f:
            json.dump(editor.repo, f, indent=2, ensure_ascii=False)
        return self.child_json_path, input_tokens, output_tokens
