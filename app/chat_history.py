"""Chat history management."""

from collections import defaultdict

# In-memory store: session_id -> list of messages
_histories: dict[str, list[dict]] = defaultdict(list)

MAX_HISTORY_TURNS = 20  # Keep last N turns (user + assistant pairs)


def add_message(session_id: str, role: str, content: str) -> None:
    """Add a message to the session history."""
    _histories[session_id].append({"role": role, "content": content})

    # Trim to keep last MAX_HISTORY_TURNS * 2 messages (user + assistant)
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_histories[session_id]) > max_msgs:
        _histories[session_id] = _histories[session_id][-max_msgs:]


def get_history(session_id: str) -> list[dict]:
    """Return the conversation history for a session."""
    return list(_histories[session_id])


def clear_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    _histories[session_id] = []


def clear_all() -> None:
    """Clear all session histories."""
    _histories.clear()
