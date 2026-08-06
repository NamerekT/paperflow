"""Text helpers shared by the LLM tasks."""


def truncate_for_llm(text: str, limit: int) -> str:
    """Cut document text down to what is worth sending to a model.

    Both the classifier and the deadline extractor used to slice `[:10000]`
    inline, in two files, with no shared name for the number. Whatever the
    limit is, it has to be the same in both places: they are sent to the same
    provider under the same token budget, and a document that fits one but not
    the other would be classified from a different span of text than the one
    its deadline was read from.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return text[:limit]
