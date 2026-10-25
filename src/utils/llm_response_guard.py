"""
Detecting truncated completions and recovering reasoning text.

vLLM's --reasoning-parser puts everything inside <think> into the `reasoning`
field (`reasoning_content` in older versions). A completion cut off while
thinking arrives with finish_reason="length", empty content and no tool calls,
which must not be mistaken for a finished answer.
"""

from typing import Any

# Below this, a file cannot hold a usable analysis or mutation plan.
MIN_USEFUL_OUTPUT_CHARS = 120


class TruncatedCompletionError(RuntimeError):
    """Raised when a completion was cut off at the token ceiling and did not recover."""


def finish_reason(response: Any) -> str:
    """The provider's finish_reason for a LangChain AIMessage, or "" if absent."""
    metadata = getattr(response, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if reason:
        return str(reason)

    extra = getattr(response, "additional_kwargs", None) or {}
    return str(extra.get("finish_reason") or "")


def was_truncated(response: Any) -> bool:
    """True when the completion hit the token ceiling instead of finishing."""
    return finish_reason(response) == "length"


def reasoning_text(response: Any) -> str:
    """The <think> content, when the server sent it and the client preserved it."""
    extra = getattr(response, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            inner = value.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
    return ""


def message_reasoning(message: Any) -> str:
    """
    The <think> content of a raw OpenAI SDK message.

    vLLM 0.25 renamed the response field from `reasoning_content` to `reasoning`
    and only accepts the old name on incoming messages, so both are checked.
    """
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def response_text(response: Any) -> str:
    """
    The answer text, falling back to the reasoning text when the answer is empty
    (with thinking disabled the parser may route the whole answer to reasoning).
    """
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    if content.strip():
        return content
    return reasoning_text(response)


def describe_response(response: Any) -> str:
    """One-line diagnosis of a response, for logs and error messages."""
    usage = getattr(response, "usage_metadata", None) or {}
    content = getattr(response, "content", "") or ""
    return " ".join([
        f"finish_reason={finish_reason(response) or '<absent>'}",
        f"content_chars={len(content)}",
        f"reasoning_chars={len(reasoning_text(response))}",
        f"tool_calls={len(getattr(response, 'tool_calls', None) or [])}",
        f"tokens={usage.get('input_tokens', 0)}in/{usage.get('output_tokens', 0)}out",
    ])


def truncation_error(agent_name: str, response: Any, max_tokens: Any = None) -> TruncatedCompletionError:
    """Raised when a model is cut off even with thinking already disabled."""
    ceiling = f"{max_tokens} tokens" if max_tokens else "the context window"
    return TruncatedCompletionError(
        f"{agent_name}: the model was truncated (finish_reason='length') even with thinking "
        f"disabled, so it spent its whole budget ({ceiling}) without producing an answer. "
        f"Response: {describe_response(response)}. "
        f"Fixes: raise META_MAX_TOKENS, or raise --max-model-len on the meta server."
    )


def empty_output_error(agent_name: str, response: Any) -> RuntimeError:
    """Raised when a loop ends with nothing worth writing to disk."""
    return RuntimeError(
        f"{agent_name}: the run finished with an EMPTY final output. Refusing to write a "
        f"report file that the next agent in the pipeline would load as if it were a real "
        f"analysis or mutation plan. Response: {describe_response(response)}."
    )


def reject_unusable_input(agent_name: str, path: str, content: str) -> None:
    """Validates a file produced by the upstream agent before using it."""
    if len(content.strip()) < MIN_USEFUL_OUTPUT_CHARS:
        raise RuntimeError(
            f"{agent_name}: the input file {path} holds only {len(content.strip())} characters "
            f"({MIN_USEFUL_OUTPUT_CHARS} is the minimum for a usable plan). The upstream agent "
            f"that wrote it produced nothing, most likely its completion was truncated "
            f"(finish_reason='length') mid-<think>. Fix the producer, not this agent. "
            f"Content: {content.strip()[:200]!r}"
        )
