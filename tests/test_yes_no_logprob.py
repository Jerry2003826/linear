"""Tests for yes/no logprob scoring using a tiny deterministic fake model.

We avoid downloading real models in unit tests. A fake tokenizer/model lets us
verify the math: score = seq_logprob(' yes') - seq_logprob(' no').
"""
import math

import torch

from linear_rag.eval.scoring import yes_no_score, answer_token_ids


class FakeTokenizer:
    # vocab: map chars/words to ids
    def __init__(self):
        self.vocab = {"<p>": 0, " yes": 1, " no": 2, "a": 3, "b": 4}

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        # split prompt chars deterministically
        return [self.vocab.get(c, 3) for c in text if c in self.vocab] or [0]


class FakeModelOut:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns fixed logits so 'yes'(id1) has higher logprob than 'no'(id2)
    at every position."""
    def __init__(self, vocab_size=5):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        T = input_ids.shape[1]
        logits = torch.zeros(1, T, self.vocab_size)
        # make token id 1 (" yes") strongly preferred everywhere
        logits[..., 1] = 5.0
        logits[..., 2] = 1.0
        return FakeModelOut(logits)


def test_answer_token_ids_nonempty():
    tok = FakeTokenizer()
    assert answer_token_ids(tok, " yes") == [1]
    assert answer_token_ids(tok, " no") == [2]


def test_yes_preferred_gives_positive_score():
    tok = FakeTokenizer()
    model = FakeModel()
    s = yes_no_score(model, tok, "ab", device="cpu",
                     yes_token=" yes", no_token=" no", max_length=512)
    assert s > 0  # yes logprob > no logprob


def test_score_matches_manual_logsoftmax():
    tok = FakeTokenizer()
    model = FakeModel()
    # manual: logprob of id1 vs id2 under softmax over [0,5,1,0,0]
    import torch.nn.functional as F
    lp = F.log_softmax(torch.tensor([0.0, 5.0, 1.0, 0.0, 0.0]), dim=-1)
    expected = float(lp[1] - lp[2])
    s = yes_no_score(model, tok, "a", device="cpu", max_length=512)
    assert abs(s - expected) < 1e-4


def test_empty_candidate_handling():
    # scoring an empty doc text should still run (prompt built upstream);
    # here we just confirm scoring doesn't crash on minimal prompt.
    tok = FakeTokenizer()
    model = FakeModel()
    s = yes_no_score(model, tok, "a", device="cpu")
    assert isinstance(s, float)
