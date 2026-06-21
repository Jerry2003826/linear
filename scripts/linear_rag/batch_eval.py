from __future__ import annotations

"""Batched yes/no scoring for causal LMs (Mamba + Pythia).

score(prompt) = logp(yes_id at next pos) - logp(no_id at next pos)

Batching strategy:
- LEFT-pad prompts to the max length in the batch so the LAST position of every
  row is a real token (the position whose next-token distribution we read).
- For Pythia (transformer) we pass attention_mask so pads are ignored.
- For Mamba (SSM, no attention_mask support in HF), left-padding still places
  real content contiguously at the end; we verify batched==single numerically
  before trusting it (verify_batch()).
Read logits at position -1 for every row.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.linear_rag.data.prompts import build_pairwise_prompt


def _encode_prompts(tok, prompts, max_len, yes_no_reserve=1):
    ids_list = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        keep = max_len - yes_no_reserve
        if len(ids) > keep:
            ids = ids[-keep:]
        ids_list.append(ids)
    return ids_list


def batch_yes_no_scores(model, tok, prompts, device, yes_id, no_id,
                        max_len=512, supports_mask=True):
    """Return np.array of (logp_yes - logp_no) for each prompt. Left-padded batch."""
    ids_list = _encode_prompts(tok, prompts, max_len)
    L = max(len(x) for x in ids_list)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attn[i, L - len(ids):] = 1
    input_ids = input_ids.to(device)
    with torch.no_grad():
        if supports_mask:
            logits = model(input_ids, attention_mask=attn.to(device)).logits
        else:
            logits = model(input_ids).logits
    last = logits[:, -1, :].float()
    lp = F.log_softmax(last, dim=-1)
    return (lp[:, yes_id] - lp[:, no_id]).cpu().numpy()


def rerank_eval_batched(model, tok, eval_qids, qmap, candidates, doc_text_map,
                        gold, device, max_len, yes_id, no_id, topk=100,
                        batch_size=32, supports_mask=True):
    """Rerank candidates for each query using batched scoring. Returns rankings + latency."""
    import time
    rankings = {}
    t0 = time.time()
    n_cand_total = 0
    for qid in eval_qids:
        q = qmap[qid]
        cand_ids = candidates.get(qid, [])[:topk]
        prompts = [build_pairwise_prompt(q["query_text"], doc_text_map[c])
                   for c in cand_ids]
        scores = []
        for i in range(0, len(prompts), batch_size):
            scores.extend(batch_yes_no_scores(
                model, tok, prompts[i:i + batch_size], device, yes_id, no_id,
                max_len, supports_mask))
        n_cand_total += len(cand_ids)
        order = np.argsort(-np.array(scores))
        rankings[qid] = [cand_ids[i] for i in order]
    dt = time.time() - t0
    return rankings, dt / max(1, len(eval_qids)), dt / max(1, n_cand_total)


def verify_batch(model, tok, prompts, device, yes_id, no_id, max_len,
                 supports_mask):
    """Compare batched vs single-prompt scores; return max abs diff."""
    from src.linear_rag.eval.scoring import yes_no_score_fast
    single = np.array([
        yes_no_score_fast(model, tok, p, device, " yes", " no", max_len)
        for p in prompts])
    batched = batch_yes_no_scores(model, tok, prompts, device, yes_id, no_id,
                                  max_len, supports_mask)
    return float(np.max(np.abs(single - batched))), single, batched
