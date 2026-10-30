"""Transplant the same attention operator while leaving new token interfaces new."""
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
from torch import nn

from src.models.vae.uv_intersection import CurveAttentionStack, UVDecoderConfig
from src.models.vae.uv_intersection_training import _curve_attention_key, _retained_key


def test_curve_attention_migration_preserves_operator_for_spatial_memory():
    torch.manual_seed(42)
    old = nn.ModuleList([
        nn.TransformerDecoderLayer(32, 4, 64, dropout=0, batch_first=True, norm_first=True)
        for _ in range(2)
    ]).eval()
    new = CurveAttentionStack(UVDecoderConfig(
        curve_width=32, curve_heads=4, curve_ffn=64, curve_layers=2, dropout=0,
    )).eval()
    source = {'decoder.inter.layers.' + key: value for key, value in old.state_dict().items()}
    mapping = {key: _curve_attention_key('decoder.intersection.curve_attention.' + key)
               for key in new.state_dict()}
    assert set(mapping.values()) == set(source)
    new.load_state_dict({key: source[old_key] for key, old_key in mapping.items()}, strict=True)
    query, memory = torch.randn(2, 16, 32), torch.randn(2, 512, 32)
    with torch.no_grad():
        expected = query
        for layer in old:
            expected = layer(expected, memory)
        torch.testing.assert_close(new(query, memory), expected, rtol=0, atol=0)
    key = 'decoder.intersection.curve_attention.blocks.0.self_attention.in_proj_weight'
    assert _retained_key(key) is None  # Historical frozen initialization is unchanged.
    assert _retained_key(key, True) == 'decoder.inter.layers.0.self_attn.in_proj_weight'
    assert _retained_key('decoder.intersection.uv_head.output.1.weight', True) is None


def test_joint_entry_unfreezes_the_whole_vae_and_requests_attention_migration():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='train_vae_uv_intersection_joint')
    assert cfg.model.freeze_encoder is False
    assert cfg.model.trainable_scope == 'all'
    assert cfg.model.init_uv_attention_from_intersection is True
    assert cfg.model.gaussian_weights == 5e-7
    assert cfg.trainer.resume_from_checkpoint is None
    assert cfg.trainer.exp_name == 'uv_intersection/joint64'
