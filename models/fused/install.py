"""Install fused decode layers on a model's backbone (and tune their kernels).

Tuning times candidate kernel configs as CUDA-graph replays, which cannot
happen inside someone else's graph capture, so every shape is exercised here
with dummy inputs before the runtime captures anything.
"""

from __future__ import annotations

import torch
from transformers import StaticCache

from .layer import FusedDecoderLayer


def install_fused_backbone(model, *, batch_size: int = 2, max_seq_len: int = 1024, mlp_bits: int | None = None) -> int:
    bb = model.backbone_model
    n = 0
    for i, layer in enumerate(bb.layers):
        if isinstance(layer, FusedDecoderLayer):
            continue
        bb.layers[i] = FusedDecoderLayer(layer, i, mlp_bits=mlp_bits)
        n += 1
    # Exercise every kernel shape once (tunes and compiles) on a scratch cache.
    dev = next(bb.parameters()).device
    dtype = next(bb.parameters()).dtype
    cache = StaticCache(config=bb.config, max_cache_len=max_seq_len, batch_size=batch_size)
    x = torch.zeros(batch_size, 1, bb.config.hidden_size, device=dev, dtype=dtype)
    mask = torch.zeros(batch_size, 1, 1, max_seq_len, device=dev, dtype=dtype)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    pos_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
    with torch.no_grad():
        cos, sin = bb.rotary_emb(x, pos_ids)
        for layer in bb.layers:
            layer(x, attention_mask=mask, position_ids=pos_ids, past_key_values=cache, use_cache=True,
                  cache_position=pos, position_embeddings=(cos, sin))
    del cache
    torch.cuda.synchronize(dev)
    return n


def install_fused_depth(model, *, batch_size: int = 2, mlp_bits: int | None = None) -> int:
    dm = model.depth_decoder.model
    n = 0
    for i, layer in enumerate(dm.layers):
        if isinstance(layer, FusedDecoderLayer):
            continue
        dm.layers[i] = FusedDecoderLayer(layer, i, mlp_bits=mlp_bits)
        n += 1
    dev = next(dm.parameters()).device
    dtype = next(dm.parameters()).dtype
    max_seq = 2 + dm.config.num_codebooks
    cache = StaticCache(config=dm.config, max_cache_len=max_seq, batch_size=batch_size)
    x = torch.zeros(batch_size, 1, dm.config.hidden_size, device=dev, dtype=dtype)
    mask = torch.zeros(batch_size, 1, 1, max_seq, device=dev, dtype=dtype)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    pos_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
    with torch.no_grad():
        cos, sin = dm.rotary_emb(x, pos_ids)
        for layer in dm.layers:
            layer(x, attention_mask=mask, position_ids=pos_ids, past_key_values=cache, use_cache=True,
                  cache_position=pos, position_embeddings=(cos, sin))
    del cache
    torch.cuda.synchronize(dev)
    return n
