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


def _from_fp8(*mods) -> FusedWeight:
    """Adopt Fp8Linear buffers (fp8 e4m3 (N, K), per-row scale (1, N)) as-is,
    concatenated along N."""
    fw = FusedWeight.__new__(FusedWeight)
    fw.bits = 8
    fw.w = torch.cat([m.weight_fp8 for m in mods], 0).contiguous()
    fw.s = torch.cat([m.weight_scale.reshape(-1) for m in mods], 0).float().contiguous()
    fw.z = None
    fw.N, fw.K = fw.w.shape
    return fw


class FusedDecoderLayer(nn.Module):
    def __init__(self, layer: nn.Module, layer_idx: int, mlp_bits: int | None = None, attn_bits: int = 16):
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
        # Attention projections stay bf16 in the stock path because _scaled_mm
        # was slower than cuBLAS on them; the fused GEMV has no such problem,
        # so attn_bits=8 halves their bytes (per-row fp8, same as the MLP).
        self.wqkv = FusedWeight(torch.cat([wq, wk, wv], 0).detach(), attn_bits)
        self.wo = FusedWeight(attn.o_proj.weight.detach(), attn_bits)
        mlp = layer.mlp
        self.tinygemm = type(mlp.gate_proj).__name__ == "Int4Linear"
        if self.tinygemm:
            self.gate, self.up, self.down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        elif type(mlp.gate_proj).__name__ == "Fp8Linear":
            self.wgu = _from_fp8(mlp.gate_proj, mlp.up_proj)
            self.wd = _from_fp8(mlp.down_proj)
        else:
            bits = mlp_bits or 16
            self.wgu = FusedWeight(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).detach(), bits)
            self.wd = FusedWeight(mlp.down_proj.weight.detach(), bits)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        layers = getattr(past_key_values, "layers", None)
        T = hidden_states.shape[1]
        if layers is None or position_embeddings is None or attention_mask is None or T > 4:
            return self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position,
                             position_embeddings=position_embeddings, **kwargs)
        if T > 1:
            # A short prefill (the depth decoder's 2-position start) is exactly
            # T causal decode steps: same rope, same mask rows, same cache.
            cos, sin = position_embeddings
            outs = []
            for t in range(T):
                outs.append(self.forward(
                    hidden_states[:, t : t + 1], attention_mask=attention_mask[:, :, t : t + 1],
                    position_ids=None if position_ids is None else position_ids[:, t : t + 1],
                    past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position[t : t + 1],
                    position_embeddings=(cos[:, t : t + 1], sin[:, t : t + 1])))
            return torch.cat(outs, dim=1)
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
