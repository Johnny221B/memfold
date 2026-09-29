"""Check warmup math and the canonical reader-initialization trainable boundary."""
import importlib.util
from pathlib import Path

import torch

from memory_opd.soft_reconstruction import ContextResampler, ContextToSoftTokens, SoftTokenProjector


def load_module(relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warmup_distinguishes_separation_alignment_and_gram():
    module = load_module('scripts/train_representation_warmup.py')
    labels = torch.tensor([0, 0, 1, 1])
    vectors = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    separation_loss, alignment_loss, gram_regularization, _ = module.all_context_losses(
        vectors, labels, maximum_cosine=0.8)
    assert separation_loss.item() == alignment_loss.item() == gram_regularization.item() == 0
    collapsed = torch.tensor([[1., 0.]]).repeat(4, 1)
    separation_loss, alignment_loss, gram_regularization, _ = module.all_context_losses(
        collapsed, labels, maximum_cosine=0.8)
    assert separation_loss > 0
    assert alignment_loss.item() == 0
    assert gram_regularization.item() == 1


def test_reader_initialization_updates_only_lora_and_final_compressor_layers():
    module = load_module('locomo_pipeline/compressor_core.py')
    policy = torch.nn.Module()
    policy.base = torch.nn.Linear(8, 8)
    policy.lora_adapter = torch.nn.Linear(8, 8)
    compressor = ContextToSoftTokens(
        ContextResampler(8, latent_dim=8, token_count=4, layers=2, heads=2),
        SoftTokenProjector(8, 8))
    module.configure_scope(policy, compressor, 'reader_initialization', joint=True)
    assert all(not p.requires_grad for p in policy.base.parameters())
    assert all(p.requires_grad for p in policy.lora_adapter.parameters())
    expected = {id(p) for part in [compressor.resampler.decoder.layers[-1],
                                  compressor.resampler.output_norm, compressor.projector]
                for p in part.parameters()}
    assert {id(p) for p in compressor.parameters() if p.requires_grad} == expected
    active = [p for part in [policy, compressor] for p in part.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(active)
    module.audit_scope(policy, compressor, optimizer, 'reader_initialization', joint=True)
