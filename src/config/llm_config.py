"""
The two model roles and how each one is served.

META: the meta-agents (Problem Understanding, Solution Proposer, Implementation).
      8,192 prompt tokens and 32,768 generation tokens.
TASK: the agent being evolved, playing ALFWorld episodes. 4,096 prompt tokens
      and 16,384 generation tokens, with a thinking-token budget that vLLM
      enforces by closing </think> once it is spent.

vLLM's --max-model-len is the total of prompt and generation, so each role
declares both halves and its server is started with their sum.

All values come from the environment (see scripts/00_config.sh). The evolving
agent cannot raise its own limits: get_task_llm() returns a client whose
invoke() takes no per-call overrides.
"""
import os
import time

from openai import OpenAI

from src.utils.llm_response_guard import message_reasoning
from src.utils.run_context import record_fallback


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# ---- meta role -------------------------------------------------------------
META_BASE_URL = os.getenv("META_BASE_URL", "http://127.0.0.1:8080/v1")
META_MODEL_ID = os.getenv("META_MODEL_ID", "qwen3.8-27b")
META_PROMPT_CONTEXT = _int("META_PROMPT_CONTEXT", 8192)
META_MAX_TOKENS = _int("META_MAX_TOKENS", 32768)
META_MAX_MODEL_LEN = _int("META_MAX_MODEL_LEN", META_PROMPT_CONTEXT + META_MAX_TOKENS)
META_THINKING = _flag("META_THINKING", True)

# ---- task role -------------------------------------------------------------
TASK_BASE_URL = os.getenv("TASK_BASE_URL", "http://127.0.0.1:8081/v1")
TASK_MODEL_ID = os.getenv("TASK_MODEL_ID", "qwen3.5-4b")
TASK_PROMPT_CONTEXT = _int("TASK_PROMPT_CONTEXT", 4096)
TASK_MAX_TOKENS = _int("TASK_MAX_TOKENS", 16384)
TASK_MAX_MODEL_LEN = _int("TASK_MAX_MODEL_LEN", TASK_PROMPT_CONTEXT + TASK_MAX_TOKENS)
TASK_THINKING = _flag("TASK_THINKING", True)
# vLLM closes </think> once this many reasoning tokens are spent
# (SamplingParams.thinking_token_budget). Ignored when TASK_THINKING is off.
TASK_THINKING_BUDGET = _int("TASK_THINKING_BUDGET", 1024)

TASK_TEMPERATURE = float(os.getenv("TASK_TEMPERATURE", "0.7"))
TASK_TOP_P = float(os.getenv("TASK_TOP_P", "0.95"))
# Retries for transient connection errors to the task server.
TASK_CONNECTION_RETRY_ATTEMPTS = int(os.getenv("TASK_CONNECTION_RETRY_ATTEMPTS", "5"))
TASK_CONNECTION_RETRY_DELAY = float(os.getenv("TASK_CONNECTION_RETRY_DELAY", "10"))
TASK_TOP_K = _int("TASK_TOP_K", 20)
TASK_MIN_P = float(os.getenv("TASK_MIN_P", "0.0"))

# Qwen recommends presence_penalty 1.5-2.0 for prose, but on ALFWorld turns
# those values made small models answer "look" instead of acting.
TASK_PRESENCE_PENALTY = float(os.getenv("TASK_PRESENCE_PENALTY", "0.0"))

# Used by TaskModelClient._fit to shrink the generation budget when a long
# episode pushes the prompt past TASK_PROMPT_CONTEXT.
CHARS_PER_TOKEN = 3.5
MIN_TASK_MAX_TOKENS = 512

LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "EMPTY")

# enable_thinking is always sent explicitly: checkpoints differ in their default,
# and vLLM's reasoning parser splits the answer based on the same flag.
NO_THINK_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def thinking_body(thinking: bool) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}


def describe_active_models() -> dict:
    """
    Identity of the models in play. Scores are only comparable across runs that
    share it, so it is stamped into the lineage archive and checked on load.
    """
    return {"backend": "local", "meta_model": META_MODEL_ID, "task_model": TASK_MODEL_ID}


def _openai_content(content):
    """Strings and OpenAI content-block lists pass through unchanged."""
    if isinstance(content, (str, list)):
        return content
    return str(content)


def _messages_to_openai(messages) -> list:
    roles = {"system": "system", "ai": "assistant"}
    return [
        {
            "role": roles.get(getattr(m, "type", "human"), "user"),
            "content": _openai_content(m.content),
        }
        for m in messages
    ]


class _SimpleResponse:
    def __init__(self, content: str, usage_metadata: dict):
        self.content = content
        self.usage_metadata = usage_metadata


def _usage(tokens_in: int, tokens_out: int) -> dict:
    return {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "input_token_details": {"cache_read": 0, "cache_creation": 0},
    }


class TaskModelClient:
    """
    The client the evolving agent plays episodes with.

    invoke(messages) is the whole contract: no keyword arguments, so a mutation
    cannot widen the context window or the completion budget from inside the
    agent. Those are set once, here, from the server's own limits.
    """

    def __init__(self):
        self.model_id = TASK_MODEL_ID
        self._client = OpenAI(api_key=LOCAL_API_KEY, base_url=TASK_BASE_URL)
        # Cleared the first time the server rejects thinking_token_budget.
        self.thinking_budget_supported = True
        self.thinking = TASK_THINKING
        self.no_think_fallbacks = 0

    @staticmethod
    def _content_tokens(content) -> float:
        """Character-based token estimate for one message's content."""
        if isinstance(content, list):
            return sum(len(block.get("text", "")) for block in content) / CHARS_PER_TOKEN
        return len(content if isinstance(content, str) else str(content)) / CHARS_PER_TOKEN

    def _fit(self, payload: list) -> int:
        """
        The generation budget that still fits beside this prompt.

        vLLM rejects prompt + max_tokens > --max-model-len, so on a late turn
        whose prompt exceeds TASK_PROMPT_CONTEXT the budget is shrunk and the
        fallback is recorded.
        """
        prompt_tokens = int(sum(self._content_tokens(m["content"]) for m in payload))
        room = TASK_MAX_MODEL_LEN - prompt_tokens
        if room >= TASK_MAX_TOKENS:
            return TASK_MAX_TOKENS
        if room < MIN_TASK_MAX_TOKENS:
            raise ValueError(
                f"the prompt is about {prompt_tokens} tokens and the server window is "
                f"{TASK_MAX_MODEL_LEN}, leaving no room to answer. Raise TASK_PROMPT_CONTEXT.")
        record_fallback("task_generation_budget_shrunk", model=self.model_id,
                        prompt_tokens=prompt_tokens, max_tokens=room)
        return room

    def _complete(self, messages, thinking: bool, budget: bool):
        payload = _messages_to_openai(messages)
        extra_body = {"top_k": TASK_TOP_K, "min_p": TASK_MIN_P}
        if thinking and budget and self.thinking_budget_supported:
            extra_body["thinking_token_budget"] = TASK_THINKING_BUDGET
        extra_body.update(thinking_body(thinking))
        return self._client.chat.completions.create(
            model=self.model_id,
            messages=payload,
            temperature=TASK_TEMPERATURE,
            top_p=TASK_TOP_P,
            presence_penalty=TASK_PRESENCE_PENALTY,
            max_tokens=self._fit(payload),
            extra_body=extra_body,
        )

    def _complete_with_retry(self, messages, thinking: bool, budget: bool):
        """Retries connection errors only; any other error is raised immediately."""
        for attempt in range(1, TASK_CONNECTION_RETRY_ATTEMPTS + 1):
            try:
                return self._complete(messages, thinking=thinking, budget=budget)
            except Exception as e:
                if "connection error" not in str(e).lower() or attempt == TASK_CONNECTION_RETRY_ATTEMPTS:
                    raise
                record_fallback("task_connection_retry", model=self.model_id,
                                attempt=attempt, error=str(e)[:300])
                print(f"  [task model] connection error (attempt {attempt}/"
                     f"{TASK_CONNECTION_RETRY_ATTEMPTS}), retrying in "
                     f"{TASK_CONNECTION_RETRY_DELAY}s: {e}")
                time.sleep(TASK_CONNECTION_RETRY_DELAY)

    @staticmethod
    def _read(response):
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return "", ""
        text = choice.message.content or ""
        if not text.strip():
            # Without a closing </think> the reasoning parser puts the whole answer
            # in the reasoning field.
            text = message_reasoning(choice.message)
        return text, getattr(choice, "finish_reason", "") or ""

    @staticmethod
    def _tokens(response):
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

    def invoke(self, messages):
        try:
            response = self._complete_with_retry(messages, thinking=self.thinking, budget=True)
        except Exception as e:
            if not self.thinking_budget_supported or "thinking_token_budget" not in str(e):
                raise
            self.thinking_budget_supported = False
            record_fallback("task_thinking_budget_rejected", model=self.model_id, error=str(e)[:300])
            print(f"  [task model] server rejected thinking_token_budget; "
                  f"continuing with unbounded thinking. ({e})")
            response = self._complete(messages, thinking=self.thinking, budget=False)

        text, finish = self._read(response)
        tokens_in, tokens_out = self._tokens(response)

        if not text.strip() and finish == "length" and self.thinking:
            self.thinking = False
            self.no_think_fallbacks += 1
            record_fallback("task_no_think", model=self.model_id, max_tokens=TASK_MAX_TOKENS)
            print(f"  [task model] completion truncated inside <think> ({TASK_MAX_TOKENS} tokens) "
                  f"; thinking disabled for the rest of this episode.")
            response = self._complete(messages, thinking=False, budget=False)
            text, _ = self._read(response)
            retry_in, retry_out = self._tokens(response)
            tokens_in += retry_in
            tokens_out += retry_out

        return _SimpleResponse(text, _usage(tokens_in, tokens_out))


def _reasoning_aware_chat_openai():
    """
    ChatOpenAI subclass that keeps reasoning_content and finish_reason.

    langchain-openai drops reasoning_content, where vLLM's --reasoning-parser
    puts the model's thinking, and finish_reason is needed to tell a completion
    truncated mid-thought from an empty answer. Defined inside a function to
    keep the langchain_openai import lazy.
    """
    from langchain_openai import ChatOpenAI

    class ReasoningAwareChatOpenAI(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            try:
                raw = response if isinstance(response, dict) else response.model_dump(
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}}, warnings=False)
                choices = raw.get("choices") or []
            except Exception:
                return result

            for generation, choice in zip(result.generations, choices):
                message = (choice or {}).get("message") or {}
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if reasoning:
                    generation.message.additional_kwargs["reasoning_content"] = reasoning
                if (choice or {}).get("finish_reason"):
                    generation.message.response_metadata.setdefault(
                        "finish_reason", choice["finish_reason"])
            return result

    return ReasoningAwareChatOpenAI


def get_meta_llm(temperature: float = 0.1, thinking: bool = None):
    """
    Tool-calling client for the meta-agents.

    thinking=False asks the chat template to close the <think> block before
    generation; src/agents/meta_loop.py keeps one client of each kind and swaps
    to the no-think one after a truncation.
    """
    thinking = META_THINKING if thinking is None else thinking
    kwargs = {
        "model": META_MODEL_ID,
        "api_key": LOCAL_API_KEY,
        "base_url": META_BASE_URL,
        "max_tokens": META_MAX_TOKENS,
        "temperature": temperature,
        "extra_body": thinking_body(thinking),
    }
    return _reasoning_aware_chat_openai()(**kwargs)


def get_task_llm():
    """The episode-playing client."""
    return TaskModelClient()
