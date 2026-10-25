"""
Checks both model servers before a run starts.

Each check targets a failure that would otherwise be silent at runtime: empty
content, tool calls that are not parsed, <think> text leaking into the answer,
an ignored thinking budget, or a context window that differs from the one the
pipeline expects.

    python scripts/preflight.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import StructuredTool
from openai import OpenAI

from src.config.llm_config import (
    META_BASE_URL, META_MAX_MODEL_LEN, META_MAX_TOKENS, META_MODEL_ID,
    META_PROMPT_CONTEXT, META_THINKING, NO_THINK_BODY, TASK_BASE_URL,
    TASK_MAX_MODEL_LEN, TASK_MAX_TOKENS, TASK_MIN_P, TASK_MODEL_ID,
    TASK_PRESENCE_PENALTY, TASK_PROMPT_CONTEXT, TASK_TEMPERATURE, TASK_THINKING,
    TASK_THINKING_BUDGET, TASK_TOP_K, TASK_TOP_P, get_meta_llm, get_task_llm, thinking_body,
)
from src.benchmarks.alfworld.alfworld_policy import build_alfworld_prompt, parse_alfworld_action
from src.utils.llm_response_guard import describe_response, message_reasoning, was_truncated

failures = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def check_meta_server():
    print("\n-- meta server --")
    client = OpenAI(api_key="EMPTY", base_url=META_BASE_URL)

    # Generous max_tokens: the model may open a <think> block first.
    reply = client.chat.completions.create(
        model=META_MODEL_ID, max_tokens=2048, temperature=0.0,
        messages=[{"role": "user", "content": "Reply exactly: OK"}],
        extra_body=thinking_body(META_THINKING))
    content = (reply.choices[0].message.content or "").strip()
    check("meta server answers a trivial prompt", bool(content), repr(content[:40]))

    check("no <think> leaking into content",
          "<think>" not in content and "</think>" not in content)

    # The meta loop's no-think fallback depends on this.
    reply = client.chat.completions.create(
        model=META_MODEL_ID, max_tokens=256, temperature=0.0,
        messages=[{"role": "user", "content": "Think step by step: what is 17 x 3?"}],
        extra_body=dict(NO_THINK_BODY))
    no_think = (reply.choices[0].message.content or "").strip()
    check("enable_thinking=False produces an answer", bool(no_think),
          f"{len(no_think)} chars, finish_reason={reply.choices[0].finish_reason}")

    check("prompt + generation fit the served window",
          META_PROMPT_CONTEXT + META_MAX_TOKENS <= META_MAX_MODEL_LEN,
          f"{META_PROMPT_CONTEXT} prompt + {META_MAX_TOKENS} generated "
          f"of {META_MAX_MODEL_LEN} served")

    # Without parsed tool_calls the meta-agents cannot inspect anything.
    def forecast(city: str) -> str:
        """Current weather in a city."""
        return f"sunny in {city}"

    tool = StructuredTool.from_function(func=forecast, name="forecast",
                                        description="Current weather in a city.")
    calls = get_meta_llm(temperature=0.0).bind_tools([tool]).invoke(
        "What is the weather in Paris? Use the forecast tool.").tool_calls
    check("tool calls are parsed", bool(calls),
          calls[0]["name"] if calls else "check --enable-auto-tool-choice/--tool-call-parser")

    # If the budget runs out inside <think>, vLLM returns finish_reason='length'
    # with empty content and no tool calls.
    probe = get_meta_llm(temperature=0.0).invoke(
        "You are analysing an agent's failure log. Think it through, then answer in one "
        "sentence: if an agent repeats the same action for 30 steps, what is the single "
        "most likely cause?")
    check("a thinking meta-agent turn completes",
          not was_truncated(probe) and bool((probe.content or "").strip()),
          describe_response(probe))


QUESTION = [{"role": "user", "content": "Think step by step: what is 17 x 3?"}]


def ask_task_server(client, thinking: bool, budget: bool):
    """One request with the same sampling parameters TaskModelClient._complete sends."""
    extra_body = {"top_k": TASK_TOP_K, "min_p": TASK_MIN_P, **thinking_body(thinking)}
    if thinking and budget:
        extra_body["thinking_token_budget"] = TASK_THINKING_BUDGET
    reply = client.chat.completions.create(
        model=TASK_MODEL_ID, messages=QUESTION, max_tokens=TASK_MAX_TOKENS,
        temperature=TASK_TEMPERATURE, top_p=TASK_TOP_P,
        presence_penalty=TASK_PRESENCE_PENALTY, extra_body=extra_body)
    choice = reply.choices[0]
    return ((choice.message.content or "").strip() or message_reasoning(choice.message).strip(),
            choice.finish_reason)


def check_task_server():
    print(f"\n-- task server (thinking {'on' if TASK_THINKING else 'off'}) --")
    client = OpenAI(api_key="EMPTY", base_url=TASK_BASE_URL)

    budget = TASK_THINKING
    try:
        answer, finish = ask_task_server(client, TASK_THINKING, budget)
    except Exception as e:
        if not budget or "thinking_token_budget" not in str(e):
            raise
        # Not fatal: the run records it as a fallback.
        print(f"[WARN] thinking_token_budget rejected: {e}")
        print("       The task model will think without a budget; every occurrence is "
              "recorded in the run's fallbacks.jsonl.")
        budget = False
        answer, finish = ask_task_server(client, TASK_THINKING, False)

    check("task server answers a trivial prompt", bool(answer),
          f"{len(answer)} chars, finish_reason={finish}"
          + (f", thinking budget {TASK_THINKING_BUDGET}" if budget else ""))

    # Without a budget, check that unbounded thinking still produces an answer.
    if TASK_THINKING and not budget:
        answer, finish = ask_task_server(client, True, False)
        if not answer:
            print(f"[WARN] unbounded thinking produced no answer (finish_reason={finish}). "
                  f"Every turn will fall back to no-think; set TASK_THINKING=off for this "
                  f"checkpoint to skip the wasted {TASK_MAX_TOKENS} tokens per turn.")

    check("prompt + generation fit the served window",
          TASK_PROMPT_CONTEXT + TASK_MAX_TOKENS <= TASK_MAX_MODEL_LEN,
          f"{TASK_PROMPT_CONTEXT} prompt + {TASK_MAX_TOKENS} generated "
          f"of {TASK_MAX_MODEL_LEN} served")

    # End to end: one turn through the genome's prompt builder, client and parser.
    from langchain_core.messages import HumanMessage

    commands = ["go to cabinet 1", "go to countertop 1", "go to sinkbasin 1", "look"]
    prompt = build_alfworld_prompt("put a mug in the sinkbasin", [
        {"turn": 0, "action": None,
         "obs": "You are in the middle of a room. Looking quickly around you, you see a "
                "cabinet 1, a countertop 1, a sinkbasin 1, and a shelf 1."}], commands)
    response = get_task_llm().invoke([HumanMessage(content=prompt)])
    action = parse_alfworld_action(response.content)

    # An empty reply fails the preflight; an illegal command only warns, since
    # that is the agent's skill, which evolution improves.
    check("a task turn produces a reply", bool((response.content or "").strip()),
          f"{len(response.content or '')} chars")
    if action in commands:
        print(f"[ OK ] the reply is a legal ALFWorld command: {action!r}")
    else:
        print(f"[WARN] the reply is not a legal command (parsed {action!r}). The baseline "
              f"may score low until the meta-agents improve the prompt.")
        print(f"       reply: {(response.content or '')[:300]!r}")


print(f"meta: {META_MODEL_ID} at {META_BASE_URL}, prompt {META_PROMPT_CONTEXT} + "
      f"generation {META_MAX_TOKENS} of {META_MAX_MODEL_LEN} served, "
      f"thinking {'on' if META_THINKING else 'off'}")
print(f"task: {TASK_MODEL_ID} at {TASK_BASE_URL}, prompt {TASK_PROMPT_CONTEXT} + "
      f"generation {TASK_MAX_TOKENS} of {TASK_MAX_MODEL_LEN} served, "
      f"thinking {f'on, budget {TASK_THINKING_BUDGET}' if TASK_THINKING else 'off'}")

check_meta_server()
check_task_server()

print("\n" + ("PREFLIGHT FAILED: " + ", ".join(failures) if failures else "PREFLIGHT OK"))
sys.exit(1 if failures else 0)
