from __future__ import annotations

import inspect

import torch

from data import TaskConfig, build_sample, make_batch, sample_to_x_labels
from models_transformer import CausalTransformerLM
from train import TrainConfig


def test_label_alignment() -> None:
    cfg = TaskConfig(n_pairs=2, n_values=4, n_keys=8, q_count=2, seed=0)
    vocab = cfg.vocab()
    full = [
        vocab.bos,
        vocab.k_tok, vocab.key_id(3), vocab.v_tok, vocab.value_id(1),
        vocab.k_tok, vocab.key_id(5), vocab.v_tok, vocab.value_id(2),
        vocab.q_tok, vocab.key_id(3), vocab.ans_tok, vocab.value_id(1),
        vocab.q_tok, vocab.key_id(5), vocab.ans_tok, vocab.value_id(2),
    ]
    x, labels = sample_to_x_labels(full, vocab)
    ans_positions = [i for i, tok in enumerate(x) if tok == vocab.ans_tok]
    assert ans_positions == [11, 15]
    assert labels[11] == vocab.value_id(1)
    assert labels[15] == vocab.value_id(2)
    for i, label in enumerate(labels):
        if i not in ans_positions:
            assert label == -100


def test_causal_mask_invariance() -> None:
    torch.manual_seed(0)
    model = CausalTransformerLM(vocab_size=32, max_seq_len=10, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    model.eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) % 32
    b = a.clone()
    pos = 4
    b[:, pos + 1 :] = torch.tensor([[11, 12, 13, 14, 15]]) % 32
    with torch.no_grad():
        la = model(a)[:, pos, :]
        lb = model(b)[:, pos, :]
    assert torch.max(torch.abs(la - lb)).item() < 1e-5


def test_no_final_position_classifier() -> None:
    import train

    source = inspect.getsource(train)
    assert "h[:, -1]" not in source
    cfg = TaskConfig(n_pairs=3, n_values=4, n_keys=8, q_count=2, seed=1)
    x, labels = make_batch(cfg, 5, "cpu")
    assert int((labels != -100).sum().item()) == 5 * 2


if __name__ == "__main__":
    test_label_alignment()
    test_causal_mask_invariance()
    test_no_final_position_classifier()
    print("TESTS OK")
