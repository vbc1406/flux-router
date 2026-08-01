"""
Context compression — trim conversation history when it exceeds available model windows.

No LLM calls.  Pure structural manipulation of message history.
Target: < 2 ms per invocation.

Strategy (applied in order until the context fits):
  1. Keep system prompt intact — never compress it.
  2. Keep the last 4 messages intact — they carry the most relevant context.
  3. For older messages:
     a. Drop tool_result / function_result messages (verbose, low signal).
     b. Truncate long assistant responses.
     c. Drop image blocks from old messages.
     d. If still too large: keep only user messages from the old history.
  4. Append a system note about the compression.
"""

from __future__ import annotations

import structlog

from .config import CONTEXT_SUMMARY_TARGET_TOKENS
from .schemas import RoutingRequest

log = structlog.get_logger(__name__)

# Number of recent messages always kept verbatim.
_PRESERVE_RECENT = 4
# When truncating a long assistant message, keep this many words from the start
# and this many from the end so the gist is retained.
_TRUNC_HEAD_WORDS = 200
_TRUNC_TAIL_WORDS = 100


def _count_tokens(text: str) -> int:
    """Same fast approximation used in the classifier (word_count × 1.3)."""
    return max(1, int(len(text.split()) * 1.3))


def _message_tokens(msg: dict) -> int:
    """Estimate token cost for a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _count_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += _count_tokens(block.get("text", ""))
                elif block.get("type") in ("image_url", "image"):
                    total += 512  # fixed overhead for image encoding
        return total
    return 0


def estimate_tokens(messages: list[dict]) -> int:
    """Total token estimate for a list of messages."""
    return sum(_message_tokens(m) for m in messages)


def _truncate_text(text: str, head: int = _TRUNC_HEAD_WORDS, tail: int = _TRUNC_TAIL_WORDS) -> str:
    """Keep first `head` and last `tail` words, replacing the middle with an ellipsis."""
    words = text.split()
    if len(words) <= head + tail:
        return text
    return " ".join(words[:head]) + " … [truncated] … " + " ".join(words[-tail:])


def _compress_old_message(msg: dict) -> dict | None:
    """
    Apply compression to a single non-recent message.
    Returns None if the message should be dropped entirely, otherwise a (possibly
    shrunk) copy of the message.
    """
    role = msg.get("role", "")

    # Drop tool/function result messages — they are usually verbose JSON blobs
    # with low marginal value once the conversation has moved on.
    if role in ("tool", "function") or msg.get("type") in ("tool_result", "function_result"):
        return None

    content = msg.get("content", "")

    if isinstance(content, str):
        if role == "assistant" and len(content.split()) > _TRUNC_HEAD_WORDS + _TRUNC_TAIL_WORDS:
            return {**msg, "content": _truncate_text(content)}
        return msg

    if isinstance(content, list):
        # Drop image blocks; keep text blocks (possibly truncated)
        new_blocks: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            if block.get("type") in ("image_url", "image"):
                continue  # drop image
            if block.get("type") == "text":
                text = block.get("text", "")
                if (
                    role == "assistant"
                    and len(text.split()) > _TRUNC_HEAD_WORDS + _TRUNC_TAIL_WORDS
                ):
                    new_blocks.append({**block, "text": _truncate_text(text)})
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)

        if not new_blocks:
            return None
        return {**msg, "content": new_blocks}

    return msg


class ContextCompressor:
    """
    Trim conversation history to fit within a target token budget.

    Always operates on a COPY of the request — the original RoutingRequest
    is immutable from the caller's perspective.
    """

    def compress(
        self,
        request: RoutingRequest,
        target_tokens: int = CONTEXT_SUMMARY_TARGET_TOKENS,
    ) -> RoutingRequest:
        """
        Return a new RoutingRequest whose message_history has been trimmed to fit
        within `target_tokens`.  The system_prompt and raw_prompt are never touched.
        """
        history = request.message_history
        total = estimate_tokens(history)

        if total <= target_tokens:
            return request  # nothing to do

        # Split into recent (preserved) and old (candidates for compression)
        recent = history[-_PRESERVE_RECENT:] if len(history) >= _PRESERVE_RECENT else history
        old = history[:-_PRESERVE_RECENT] if len(history) > _PRESERVE_RECENT else []

        dropped = 0

        # Pass 1: compress / drop old messages
        compressed_old: list[dict] = []
        for msg in old:
            result = _compress_old_message(msg)
            if result is None:
                dropped += 1
            else:
                compressed_old.append(result)

        new_history = compressed_old + recent
        current_tokens = estimate_tokens(new_history)

        # Pass 2: if still too large, keep only user messages from old history
        if current_tokens > target_tokens:
            user_only = [m for m in compressed_old if m.get("role") == "user"]
            dropped += len(compressed_old) - len(user_only)
            compressed_old = user_only
            new_history = compressed_old + recent
            current_tokens = estimate_tokens(new_history)

        # Pass 3: last resort — drop oldest messages one by one. Tracks a
        # running token total and a cursor instead of pop(0) + a full
        # re-concat + re-scan of new_history on every iteration (which made
        # this O(n^2) on long histories — the common case for a multi-step
        # agent that resends its full history each step).
        if current_tokens > target_tokens and compressed_old:
            recent_tokens = estimate_tokens(recent)
            old_tokens = [_message_tokens(m) for m in compressed_old]
            running = sum(old_tokens)
            cut = 0
            while running + recent_tokens > target_tokens and cut < len(compressed_old):
                running -= old_tokens[cut]
                cut += 1
                dropped += 1
            if cut:
                compressed_old = compressed_old[cut:]
            new_history = compressed_old + recent
            current_tokens = running + recent_tokens

        # Append compression note so the model knows context was trimmed
        if dropped:
            note = {
                "role": "system",
                "content": (
                    f"[Context compressed. {dropped} older message(s) were trimmed "
                    f"to fit model context limits.]"
                ),
            }
            new_history = [note] + new_history

        log.debug(
            "context_compressed",
            original_tokens=total,
            compressed_tokens=current_tokens,
            messages_dropped=dropped,
        )

        # Return a copy with the new history
        return request.model_copy(update={"message_history": new_history})
