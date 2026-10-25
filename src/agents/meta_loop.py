"""
The tool-calling loop shared by the three meta-agents (on-demand retrieval).

Handles truncation recovery, context growth, repeated tool calls, running out
of context window, and selecting the final answer. The full conversation is
written to the agent's trace file.
"""
import json
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config.llm_config import (
    META_MAX_MODEL_LEN, META_MAX_TOKENS, META_THINKING, get_meta_llm,
)
from src.utils.llm_response_guard import (
    MIN_USEFUL_OUTPUT_CHARS, describe_response, reasoning_text, response_text,
    truncation_error, was_truncated,
)
from src.utils.run_context import record_fallback

MAX_TOOL_RESULT_CHARS = 8000
MAX_REPEATED_CALLS = 3
# Token estimation errs high: JSON, code and logs tokenize worse than prose.
CHARS_PER_TOKEN = 3.0
# Cost of the bound tool schemas, used until the server reports a real count.
TOOL_SCHEMA_TOKENS = 2048
COLD_SCALE = 1.2
GROWTH_SCALE = 1.4
SAFETY_TOKENS = 6144
MIN_GENERATION = 4096
MAX_OVERFLOW_RETRIES = 6
DIGEST_CHARS = 20000
BRIEFING_CHARS = 4000
ELIDED = "[elided to fit the context window — re-read this file if you still need it]"

FINAL_ANSWER_INSTRUCTION = """
You have run out of context window and can make NO further tool calls. Everything
you gathered is summarised above.

Write your FINAL ANSWER NOW, in exactly the format the system prompt requires,
using only the evidence above. Do not ask for more data and do not describe what
you would have looked at next: report what the evidence you already have supports.
"""


def _estimate_tokens(messages) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", "") or ""
        total += len(content if isinstance(content, str) else str(content))
        for call in getattr(message, "tool_calls", None) or []:
            total += len(json.dumps(call.get("args", {})))
    return int(total / CHARS_PER_TOKEN)


def _is_context_overflow(error) -> bool:
    text = str(error).lower()
    return "maximum context length" in text or "reduce the length of the input" in text


class ContextLengthError(RuntimeError):
    """
    Raised by DirectAnswer when a single-prompt request exceeds the meta window.
    Recorded as a context_length_error outcome rather than retried.
    """


def _free_space(messages) -> bool:
    """
    Elides one block of the prompt and reports whether anything was freed.

    Tool results are elided first, oldest to newest, then long assistant turns.
    Contents are replaced rather than removed, because the server rejects tool
    calls without their matching ToolMessages.
    """
    for message in messages:
        if isinstance(message, ToolMessage) and message.content != ELIDED:
            message.content = ELIDED
            return True
    for message in messages:
        content = getattr(message, "content", "") or ""
        # The ELIDED suffix marks a turn as already reclaimed.
        if (isinstance(message, AIMessage) and isinstance(content, str)
                and len(content) > 400 and not content.endswith(ELIDED)):
            message.content = content[:400] + " " + ELIDED
            return True
    return False


class MetaLoop:
    def __init__(self, name: str, tools: list, temperature: float, trace_path: str,
                 max_steps: int = 100, final_answer_fallback: bool = True,
                 token_budget: int = None):
        self.name = name
        self.max_steps = max_steps
        # Token cap for this loop, derived from the run's --max_tokens. None disables it.
        self.token_budget = token_budget
        # Disabled for agents whose output is a tool side effect (Implementation).
        self.final_answer_fallback = final_answer_fallback
        self.trace_path = trace_path
        self.tools_map = {t.name: t for t in tools}
        self.thinking_llm = get_meta_llm(temperature=temperature).bind_tools(tools)
        self.no_think_llm = get_meta_llm(temperature=temperature, thinking=False).bind_tools(tools)
        # No tools bound, so the final-answer fallback cannot make another call.
        self.answer_llm = get_meta_llm(temperature=temperature, thinking=False)
        self.thinking = META_THINKING
        self.no_think_fallbacks = 0
        self.steps_taken = 0
        self.input_tokens = 0
        self.output_tokens = 0

        # Halved on every overflow retry and restored after a successful step.
        self.generation_cap = META_MAX_TOKENS
        # Estimate and real token count of the last request; see _prompt_tokens.
        self._sent_raw_est = None
        self._sent_real = None

        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(f"{name} TRACE: thinking={'on' if self.thinking else 'off'}, "
                    f"window={META_MAX_MODEL_LEN}, generation ceiling={META_MAX_TOKENS}\n\n")

    def _trace(self, text: str) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _disable_thinking(self, step: int, generation: int) -> None:
        self.thinking = False
        self.no_think_fallbacks += 1
        record_fallback("meta_no_think", agent=self.name, step=step, max_tokens=generation)
        self._trace(f"\n[NO-THINK FALLBACK] engaged at step {step} ({generation} tokens)\n")

    def _prompt_tokens(self, messages) -> int:
        """
        Estimated prompt size of the next request.

        Anchored on the server's token count for the previous request, so only
        the change since then is estimated. Growth is scaled up and freed space
        is not, so the estimate errs high.
        """
        raw = _estimate_tokens(messages)
        if self._sent_real is None:
            return int(raw * COLD_SCALE) + TOOL_SCHEMA_TOKENS

        delta = raw - self._sent_raw_est
        return self._sent_real + int(delta * GROWTH_SCALE if delta > 0 else delta)

    def _generation_budget(self, messages):
        """
        Elides context until the prompt leaves room to answer, and returns the
        generation budget for this turn, or None when it cannot be made to fit.
        """
        while True:
            room = META_MAX_MODEL_LEN - self._prompt_tokens(messages) - SAFETY_TOKENS
            if room >= MIN_GENERATION:
                return min(META_MAX_TOKENS, self.generation_cap, room)
            if not _free_space(messages):
                return None

    def _run_tool(self, call: dict) -> str:
        name, args = call["name"], call.get("args") or {}
        if isinstance(args, dict):
            args.pop("self", None)
        self._trace(f"[Tool call] {name}\n[Args] {json.dumps(args, indent=2)}\n")

        tool = self.tools_map.get(name)
        if tool is None:
            return f"Error: tool '{name}' does not exist. Available: {list(self.tools_map)}"
        result = str(tool.invoke(args))
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
        return result

    @staticmethod
    def _digest(messages: list) -> str:
        """
        A compressed digest of the conversation for the final-answer fallback.
        Newest content is kept first; elided content is skipped.
        """
        blocks, budget = [], DIGEST_CHARS

        for message in reversed(messages[2:]):
            content = getattr(message, "content", "") or ""
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()

            if isinstance(message, ToolMessage):
                label = "TOOL RESULT"
            elif isinstance(message, AIMessage):
                label = "YOUR NOTES"
                calls = ", ".join(
                    f"{c['name']}({json.dumps(c.get('args') or {})[:200]})"
                    for c in message.tool_calls or [])
                if calls:
                    blocks.append(f"[TOOLS YOU CALLED] {calls}")
                    budget -= len(blocks[-1])
            else:
                continue

            if not content or content.endswith(ELIDED):
                continue
            if budget <= 0:
                break
            blocks.append(f"[{label}]\n{content[-budget:]}")
            budget -= len(blocks[-1])

        return "\n\n".join(reversed(blocks))

    def _final_answer(self, messages: list, reason: str):
        """
        Asks for the final answer from a compact prompt (system prompt, briefing
        and a digest of the evidence gathered), with no tools and thinking off.
        Returns None if this is disabled or produces nothing usable.
        """
        if not self.final_answer_fallback:
            return None

        record_fallback("meta_context_exhausted_final_answer", agent=self.name,
                        step=self.steps_taken, reason=reason)
        self._trace(f"\n[CONTEXT EXHAUSTED] {reason}\n"
                    f"[FINAL ANSWER FALLBACK] retrying without tools from a digest\n")

        briefing = getattr(messages[1], "content", "") if len(messages) > 1 else ""
        if not isinstance(briefing, str):
            briefing = str(briefing)

        compressed = [
            messages[0],
            HumanMessage(content=briefing[:BRIEFING_CHARS]),
            HumanMessage(content=f"WHAT YOU HAVE GATHERED SO FAR:\n\n{self._digest(messages)}\n\n"
                                 f"{FINAL_ANSWER_INSTRUCTION}"),
        ]

        try:
            response = self.answer_llm.bind(max_tokens=MIN_GENERATION).invoke(compressed)
        except Exception as e:
            self._trace(f"[FINAL ANSWER FALLBACK] failed: {e}\n")
            return None

        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self._trace(f"\n--- Final answer fallback ({describe_response(response)}) ---\n"
                    f"[Content]\n{response_text(response)}\n\n")

        if len(response_text(response).strip()) < MIN_USEFUL_OUTPUT_CHARS:
            self._trace("[FINAL ANSWER FALLBACK] produced nothing usable\n")
            return None
        return response

    def run(self, messages: list) -> AIMessage:
        """
        Drives the conversation to a final answer and returns the last AIMessage.
        Raises if the agent never produced one.
        """
        last_ai = last_signature = failure = None
        repeats = overflows = 0

        step = 0
        while step < self.max_steps:
            step += 1
            if self.token_budget is not None and (self.input_tokens + self.output_tokens) >= self.token_budget:
                answer = self._final_answer(
                    messages, f"reached its {self.token_budget:,}-token budget at step {step}")
                if answer is not None:
                    return answer
                failure = RuntimeError(
                    f"reached its {self.token_budget:,}-token budget at step {step}, and the "
                    "final-answer fallback produced nothing usable")
                self._trace(f"\n[TOKEN BUDGET] {failure}\n")
                break
            raw_est = _estimate_tokens(messages)
            generation = self._generation_budget(messages)
            if generation is None:
                answer = self._final_answer(messages, f"no room left to answer at step {step}")
                if answer is not None:
                    return answer
                failure = RuntimeError(
                    f"the prompt no longer leaves {MIN_GENERATION} tokens to answer in within "
                    f"the {META_MAX_MODEL_LEN}-token window, and the final-answer fallback "
                    f"produced nothing usable")
                self._trace(f"\n[STEP ERROR] {failure}\n")
                break

            client = self.thinking_llm if self.thinking else self.no_think_llm
            try:
                response = client.bind(max_tokens=generation).invoke(messages)
            except Exception as e:
                # Overflow: free context and halve the generation cap, then retry.
                if _is_context_overflow(e) and overflows < MAX_OVERFLOW_RETRIES:
                    freed = _free_space(messages)
                    self.generation_cap = max(MIN_GENERATION,
                                              min(self.generation_cap, generation) // 2)
                    overflows += 1
                    record_fallback("meta_context_overflow_retry", agent=self.name, step=step,
                                    attempt=overflows, generation_cap=self.generation_cap)
                    self._trace(f"\n[CONTEXT OVERFLOW] retry {overflows} at step {step} "
                                f"(cap now {self.generation_cap}, freed={freed}): {e}\n")
                    step -= 1
                    continue

                if _is_context_overflow(e):
                    answer = self._final_answer(messages, f"overflowed at step {step}: {e}")
                    if answer is not None:
                        return answer
                failure = e
                self._trace(f"\n[STEP ERROR] {e}\n")
                break

            messages.append(response)
            usage = getattr(response, "usage_metadata", None) or {}
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            if usage.get("input_tokens"):
                self._sent_raw_est, self._sent_real = raw_est, usage["input_tokens"]
            self.generation_cap = META_MAX_TOKENS

            self._trace(f"\n--- Step {step} ({describe_response(response)}) ---\n")
            if reasoning_text(response):
                self._trace(f"[Reasoning]\n{reasoning_text(response)}\n\n")
            if response.content:
                self._trace(f"[Content]\n{response.content}\n\n")

            if was_truncated(response):
                self._trace(f"[TRUNCATED] {describe_response(response)}\n")
                # Drop the turn: it may carry a half-emitted tool call.
                messages.pop()
                if self.thinking:
                    self._disable_thinking(step, generation)
                    continue
                raise truncation_error(self.name, response, generation)

            last_ai = response
            self.steps_taken = step
            if not response.tool_calls:
                self._trace("\n[Finished]\n")
                return self._finish(last_ai, failure)

            signature = json.dumps(
                [[c["name"], c.get("args")] for c in response.tool_calls], sort_keys=True)
            repeats = repeats + 1 if signature == last_signature else 0
            last_signature = signature

            for call in response.tool_calls:
                if repeats >= MAX_REPEATED_CALLS - 1:
                    result = (f"Error: you have called this exact tool with these exact arguments "
                              f"{repeats + 1} times and the result will not change. Call a "
                              f"DIFFERENT tool, or give your final answer now.")
                    self._trace(f"[Repeat guard] {result}\n\n")
                else:
                    result = self._run_tool(call)
                    self._trace(f"[Tool result]\n{result}\n\n")
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # Out of steps while still calling tools: ask for the final answer.
        if failure is None and last_ai is not None and last_ai.tool_calls:
            answer = self._final_answer(messages, f"reached the {self.max_steps}-step limit")
            if answer is not None:
                return answer

        return self._finish(last_ai, failure)

    def _finish(self, last_ai, failure):
        self._trace(f"\n[no-think fallbacks] {self.no_think_fallbacks}/{self.steps_taken}\n")
        if last_ai is None or (failure is not None and last_ai.tool_calls):
            raise RuntimeError(
                f"{self.name} ended without a final answer after {self.steps_taken} steps. "
                f"Last error: {failure}")
        return last_ai


class DirectAnswer:
    """
    Single-prompt meta-agent driver (--meta_mode direct).

    All context is inlined into one prompt and answered with one call. A
    truncated generation is retried once with thinking off; a prompt that
    exceeds the window raises ContextLengthError.
    """

    def __init__(self, name: str, temperature: float, trace_path: str):
        self.name = name
        self.trace_path = trace_path
        self.thinking_llm = get_meta_llm(temperature=temperature)
        self.no_think_llm = get_meta_llm(temperature=temperature, thinking=False)
        self.thinking = META_THINKING
        self.no_think_fallbacks = 0
        self.input_tokens = 0
        self.output_tokens = 0

        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(f"{name} TRACE (direct mode): thinking={'on' if self.thinking else 'off'}, "
                    f"window={META_MAX_MODEL_LEN}, generation ceiling={META_MAX_TOKENS}\n\n")

    def _trace(self, text: str) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def run(self, messages: list) -> AIMessage:
        """One-shot answer. Raises ContextLengthError on prompt overflow."""
        client = self.thinking_llm if self.thinking else self.no_think_llm
        try:
            response = client.bind(max_tokens=META_MAX_TOKENS).invoke(messages)
        except Exception as e:
            if _is_context_overflow(e):
                estimate = _estimate_tokens(messages)
                record_fallback("meta_context_length_error", agent=self.name, mode="direct",
                                estimated_prompt_tokens=estimate, limit=META_MAX_MODEL_LEN,
                                error=str(e)[:300])
                self._trace(f"\n[CONTEXT LENGTH ERROR] prompt (~{estimate} tokens) exceeds the "
                            f"{META_MAX_MODEL_LEN}-token window in direct mode, no recovery "
                            f"attempted: {e}\n")
                print(f"  [META CONTEXT LENGTH ERROR] {self.name}: prompt exceeds the "
                      f"{META_MAX_MODEL_LEN}-token window in direct mode (no recovery attempted)")
                raise ContextLengthError(
                    f"{self.name}: prompt (~{estimate} tokens) exceeds the {META_MAX_MODEL_LEN}"
                    f"-token window in direct mode: {e}") from e
            raise

        self._record_usage(response)
        self._trace(f"--- Response ({describe_response(response)}) ---\n")
        if reasoning_text(response):
            self._trace(f"[Reasoning]\n{reasoning_text(response)}\n\n")
        if response.content:
            self._trace(f"[Content]\n{response.content}\n\n")

        if was_truncated(response) and self.thinking:
            self.thinking = False
            self.no_think_fallbacks += 1
            record_fallback("meta_no_think", agent=self.name, mode="direct",
                            max_tokens=META_MAX_TOKENS)
            self._trace("\n[NO-THINK FALLBACK] retrying once with thinking off\n")
            response = self.no_think_llm.bind(max_tokens=META_MAX_TOKENS).invoke(messages)
            self._record_usage(response)
            self._trace(f"--- Retry response ({describe_response(response)}) ---\n"
                        f"[Content]\n{response.content}\n\n")

        if was_truncated(response):
            raise truncation_error(self.name, response, META_MAX_TOKENS)

        return response
