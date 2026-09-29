import json

import pytest
import torch

from memory_opd.soft_reconstruction import (
    ContextResampler,
    ContextToSoftTokens,
    SoftTokenProjector,
    cross_context_separation_loss,
    different_context_example,
    load_reconstruction_examples,
    load_text_memory_examples,
    reconstruction_prompt,
    serialize_memory,
    split_examples_by_context,
)


def test_context_to_soft_tokens_has_fixed_shape_and_gradients():
    module = ContextToSoftTokens(
        ContextResampler(24, latent_dim=16, token_count=5, layers=2, heads=4),
        SoftTokenProjector(16, 32),
    )
    states = torch.randn(3, 11, 24)
    mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4, [1] * 4 + [0] * 7])
    output = module(states, mask)
    assert output.shape == (3, 5, 32)
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())


def test_context_residual_explicitly_changes_queries_with_context():
    torch.manual_seed(7)
    module = ContextResampler(
        8, latent_dim=8, token_count=3, layers=1, heads=2, context_residual=True
    ).eval()
    states = torch.stack((torch.ones(5, 8), -torch.ones(5, 8)))
    mask = torch.ones(2, 5, dtype=torch.long)

    output = module(states, mask)

    assert module.context_gate.item() == pytest.approx(1.0)
    assert not torch.allclose(output[0], output[1])


def test_loader_uses_legal_prefix_and_does_not_expose_answer_or_options(tmp_path):
    questions = tmp_path / "questions.jsonl"
    contexts = tmp_path / "contexts.jsonl"
    memories = tmp_path / "memories.jsonl"
    questions.write_text(json.dumps({
        "question_id": "q1", "shared_context_id": "c1", "question": "What changed?",
        "options": ["SECRET_A", "B", "C", "D"], "answer": "(a)",
        "end_index_in_shared_context": -1,
    }) + "\n")
    contexts.write_text(json.dumps({"c1": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "future"},
    ]}) + "\n")
    memories.write_text(json.dumps({
        "question_id": "q1", "memory": {
            "evidence": ["first"], "temporal_relations": [], "derived_facts": []
        }
    }) + "\n")
    examples, state_text = load_reconstruction_examples(questions, contexts, memories)
    assert len(examples) == 1
    assert examples[0].prefix_end == 2
    assert "future" not in state_text[examples[0].state_id]
    serialized = json.dumps(examples[0].__dict__)
    assert "SECRET_A" not in serialized and "(a)" not in serialized
    assert "What changed?" in reconstruction_prompt(examples[0].question)



def test_text_memory_loader_uses_memory_as_state_and_never_loads_context(tmp_path):
    questions = tmp_path / "questions.jsonl"
    memories = tmp_path / "memories.jsonl"
    questions.write_text(json.dumps({
        "question_id": "q1", "shared_context_id": "c1", "question": "What changed?",
        "options": ["SECRET_A", "B", "C", "D"], "answer": "(a)",
        "end_index_in_shared_context": 99,
    }) + "\n")
    memories.write_text(json.dumps({
        "question_id": "q1", "memory": {
            "evidence": ["memory fact"], "temporal_relations": [], "derived_facts": []
        }
    }) + "\n")

    examples, state_text = load_text_memory_examples(questions, memories)

    assert len(examples) == 1
    assert examples[0].prefix_end == -1
    assert examples[0].state_id.startswith("memory-q1-")
    assert state_text[examples[0].state_id] == examples[0].memory_text
    assert "memory fact" in state_text[examples[0].state_id]
    assert "SECRET_A" not in json.dumps(examples[0].__dict__)

def test_split_holds_out_complete_contexts():
    from memory_opd.soft_reconstruction import ReconstructionExample

    examples = [
        ReconstructionExample(f"q{i}", f"c{i // 2}", i, f"s{i}", "q", "m")
        for i in range(8)
    ]
    train, validation = split_examples_by_context(examples, validation_contexts=1)
    assert {x.context_id for x in train}.isdisjoint({x.context_id for x in validation})


def test_negative_example_comes_from_a_different_context():
    from memory_opd.soft_reconstruction import ReconstructionExample

    examples = [
        ReconstructionExample("q1", "c1", 1, "s1", "q", "m"),
        ReconstructionExample("q2", "c1", 2, "s2", "q", "m"),
        ReconstructionExample("q3", "c2", 1, "s3", "q", "m"),
    ]
    assert different_context_example(examples, 0).context_id == "c2"
    with pytest.raises(ValueError, match="two distinct contexts"):
        different_context_example(examples[:2], 0)


def test_memory_schema_is_strict():
    assert serialize_memory({
        "evidence": ["a"], "temporal_relations": [], "derived_facts": ["b"]
    }).startswith('{"evidence"')
    with pytest.raises(ValueError, match="exactly"):
        serialize_memory({"evidence": [], "temporal_relations": [], "derived_facts": [], "x": []})


def test_cross_context_separation_penalizes_collapsed_soft_memories():
    torch.manual_seed(11)
    own = torch.randn(2, 5, 8, requires_grad=True)
    collapsed = (own.detach() + 1e-3 * torch.randn_like(own)).requires_grad_(True)

    loss, cosine = cross_context_separation_loss(
        own, collapsed, maximum_cosine=0.8
    )

    assert cosine.item() > 0.999
    assert 0.5 < loss.item() < 0.7
    loss.backward()
    assert own.grad is not None and own.grad.abs().sum() > 0
    assert collapsed.grad is not None and collapsed.grad.abs().sum() > 0


def test_cross_context_separation_accepts_already_distinct_memories():
    own = torch.ones(1, 3, 4)
    negative = -torch.ones(1, 3, 4)

    loss, cosine = cross_context_separation_loss(
        own, negative, maximum_cosine=0.8
    )

    assert cosine.item() == pytest.approx(-1.0)
    assert loss.item() == 0.0
