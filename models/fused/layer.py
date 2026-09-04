"""Fused decode-step forward for a decoder layer (batch of CFG rows, seq 1).

Wraps a transformers Qwen3DecoderLayer (the backbone) or a BreezeDecoderLayer
(the depth decoder). Any call that is not a single-token decode step against
a StaticCache goes to the original layer, so prefill and eager paths are
untouched. Installed after quantisation: an Int4Linear MLP keeps its
tinygemm matmuls (they are faster than the fused kernel at 4 bits) with the
norm, silu*up and residual fused around them; bf16 and fp8 MLPs go through
rms_gemv with the epilogues.
"""

from __future__ import annotations

import torch
from torch import nn

from .kernels import FusedWeight, attention_step, rms_gemv, rmsnorm, silu_mul


class FusedDecoderLayer(nn.Module):
    def __init__(self, layer: nn.Module, layer_idx: int, mlp_bits: int | None = None):
        super().__init__()
        self.orig = layer
        self.layer_idx = layer_idx
        attn = layer.self_attn
        self.eps = float(layer.input_layernorm.variance_epsilon)
        self.hkv = int(attn.config.num_key_value_heads)
        self.hd = int(attn.head_dim)
        self.ln1 = layer.input_layernorm.weight
        self.ln2 = layer.post_attention_layernorm.weight
        self.qn = getattr(attn, "q_norm", None).weight if getattr(attn, "q_norm", None) is not None else None
        self.kn = getattr(attn, "k_norm", None).weight if getattr(attn, "k_norm", None) is not None else None
        wq, wk, wv = attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
        self.wqkv = FusedWeight(torch.cat([wq, wk, wv], 0).detach(), 16)
        self.wo = FusedWeight(attn.o_proj.weight.detach(), 16)
        mlp = layer.mlp
        self.tinygemm = type(mlp.gate_proj).__name__ == "Int4Linear"
        if self.tinygemm:
            self.gate, self.up, self.down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        else:
            bits = mlp_bits or 16
            self.wgu = FusedWeight(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).detach(), bits)
            self.wd = FusedWeight(mlp.down_proj.weight.detach(), bits)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        layers = getattr(past_key_values, "layers", None)
        if hidden_states.shape[1] != 1 or layers is None or position_embeddings is None or attention_mask is None:
            return self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position,
                             position_embeddings=position_embeddings, **kwargs)
        layer_cache = layers[self.layer_idx]
        if not layer_cache.is_initialized:
            layer_cache.lazy_initialization(hidden_states.new_zeros(hidden_states.shape[0], self.hkv, 1, self.hd))
        x = hidden_states[:, 0]
        qkv = rms_gemv(x, self.wqkv, self.ln1, self.eps)
        cos, sin = position_embeddings
        attn = attention_step(qkv, cos, sin, cache_position, layer_cache.keys, layer_cache.values, attention_mask,
                              self.qn, self.kn, self.eps)
        h = rms_gemv(attn, self.wo, None, epilogue=2, residual=x)
        if self.tinygemm:
            normed = rmsnorm(h, self.ln2, self.eps)
            act = silu_mul(self.gate(normed), self.up(normed))
            out = self.down(act) + h
        else:
            act = rms_gemv(h, self.wgu, self.ln2, self.eps, epilogue=1)
            out = rms_gemv(act, self.wd, None, epilogue=2, residual=h)
        return out.unsqueeze(1)
