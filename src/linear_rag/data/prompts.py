from __future__ import annotations

# Prompt format A: pairwise yes/no relevance (R2 zero-shot, R3 training).
PAIRWISE_YESNO_TEMPLATE = (
    "Query:\n{query_text}\n"
    "Document:\n{doc_text}\n"
    "Is this document relevant to the query? Answer yes or no."
)

YES_TOKEN = " yes"
NO_TOKEN = " no"


def build_pairwise_prompt(query_text: str, doc_text: str) -> str:
    """Build the pairwise yes/no relevance prompt.

    Order is fixed: Query first, then Document. Tests assert this ordering.
    """
    if query_text is None or doc_text is None:
        raise ValueError("query_text and doc_text must not be None")
    return PAIRWISE_YESNO_TEMPLATE.format(
        query_text=query_text, doc_text=doc_text
    )
