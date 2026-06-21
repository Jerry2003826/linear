from __future__ import annotations

"""Yes/No relevance scoring for causal LMs.

score = seq_logprob(prompt + " yes") - seq_logprob(prompt + " no")
where seq_logprob sums the per-token logprobs of the answer tokens only
(conditioned on the prompt). Handles multi-token answers correctly.
"""

from typing import List


def answer_token_ids(tokenizer, answer: str) -> List[int]:
    """Token ids for an answer string like ' yes'. Handles leading-space tokens.
    Falls back to encoding without special tokens.
    """
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) == 0:
        # some tokenizers strip leading space; try alternate
        ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
    return ids


def sequence_logprob_for_answer(model, tokenizer, prompt: str, answer: str,
                                device, max_length: int = 512) -> float:
    """Sum of log p(answer_tokens | prompt) under a causal LM."""
    import torch
    import torch.nn.functional as F

    ans_ids = answer_token_ids(tokenizer, answer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # truncate prompt from the left to keep room for the answer
    keep = max_length - len(ans_ids)
    if keep > 0 and len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:]
    input_ids = torch.tensor([prompt_ids + ans_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    # token at position t is predicted by logits at position t-1
    total = 0.0
    start = len(prompt_ids)
    for j, tok in enumerate(ans_ids):
        pos = start + j - 1  # logits index that predicts this token
        total += float(logprobs[0, pos, tok].item())
    return total


def yes_no_score(model, tokenizer, prompt: str, device,
                 yes_token: str = " yes", no_token: str = " no",
                 max_length: int = 512) -> float:
    yes_lp = sequence_logprob_for_answer(
        model, tokenizer, prompt, yes_token, device, max_length
    )
    no_lp = sequence_logprob_for_answer(
        model, tokenizer, prompt, no_token, device, max_length
    )
    return yes_lp - no_lp
