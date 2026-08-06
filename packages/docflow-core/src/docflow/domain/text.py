"""Text helpers shared by the LLM tasks."""


def truncate_for_llm(text: str, limit: int) -> str:
    """Shared limit: classifier and deadline extractor must truncate identically, or a
    document would be classified from a different span of text than its deadline was read from."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return text[:limit]
