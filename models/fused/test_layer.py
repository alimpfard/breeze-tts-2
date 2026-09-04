"""Fused decode layer vs the original, on the real backbone, same cache state.

    python -m models.fused.test_layer /path/to/breeze --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import StaticCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from breeze_infer.runtime import load_runtime
from models.fused.kernels import graph_time_us
from models.fused.layer import FusedDecoderLayer
from models.int4_linear import quantize_module_int4


def run_layer(layer, model, cache, steps, B, dev, dtype):
    """Feed random hidden states one step at a time; return outputs."""
    torch.manual_seed(0)
    outs = []
    S = cache.max_cache_len
    mask = torch.full((B, 1, 1, S), torch.finfo(dtype).min, device=dev, dtype=dtype)
    for t in range(steps):
        x = torch.randn(B, 1, model.config.hidden_size, device=dev, dtype=dtype) * 0.5
        pos = torch.tensor([t], device=dev)
        pos_ids = torch.full((B, 1), t, device=dev)
        mask[:, :, :, : t + 1] = 0.0
        cos, sin = model.rotary_emb(x, pos_ids)
        y = layer(x, attention_mask=mask, position_ids=pos_ids, past_key_values=cache, use_cache=True,
                  cache_position=pos, position_embeddings=(cos, sin))
        y = y[0] if isinstance(y, tuple) else y
        outs.append(y.float())
    return torch.stack(outs), (x, mask, pos_ids, pos, (cos, sin))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("breeze", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seq", type=int, default=256)
    args = p.parse_args()
    dev = args.device
    _, model, _ = load_runtime(args.breeze, device=dev, attn_implementation="eager")
    from models.fp8_linear import quantize_module_fp8

    B = 2
    targets = [("backbone", model.backbone_model, args.layers, args.seq, ("bf16", "int4-tinygemm")),
               ("depth", model.depth_decoder.model, [0, 5, 11], 18, ("bf16", "fp8"))]
    for name, bb, layer_ids, seq, variants in targets:
      dtype = next(bb.parameters()).dtype
      for variant in variants:
        for li in layer_ids:
            import copy
            layer = copy.deepcopy(bb.layers[li])
            if variant.startswith("int4"):
                quantize_module_int4(layer)
            elif variant == "fp8":
                quantize_module_fp8(layer)
            fused = FusedDecoderLayer(layer, li)
            c1 = StaticCache(config=bb.config, max_cache_len=seq, batch_size=B)
            c2 = StaticCache(config=bb.config, max_cache_len=seq, batch_size=B)
            steps = min(args.steps, seq - 2)
            ref, last = run_layer(layer, bb, c1, steps, B, dev, dtype)
            out, _ = run_layer(fused, bb, c2, steps, B, dev, dtype)
            if name == "depth":
                # the graph's 2-position prefill: one call with T=2 on fresh caches
                c3 = StaticCache(config=bb.config, max_cache_len=seq, batch_size=B)
                c4 = StaticCache(config=bb.config, max_cache_len=seq, batch_size=B)
                x2 = torch.randn(B, 2, bb.config.hidden_size, device=dev, dtype=dtype) * 0.5
                pos2 = torch.tensor([0, 1], device=dev)
                pid2 = pos2.unsqueeze(0).expand(B, 2)
                m2 = torch.full((B, 1, 2, seq), torch.finfo(dtype).min, device=dev, dtype=dtype)
                m2[:, :, 0, :1] = 0
                m2[:, :, 1, :2] = 0
                cs2 = bb.rotary_emb(x2, pid2)
                r2 = layer(x2, attention_mask=m2, position_ids=pid2, past_key_values=c3, use_cache=True, cache_position=pos2, position_embeddings=cs2)
                r2 = r2[0] if isinstance(r2, tuple) else r2
                o2 = fused(x2, attention_mask=m2, position_ids=pid2, past_key_values=c4, use_cache=True, cache_position=pos2, position_embeddings=cs2)
                perr = ((r2.float() - o2.float()).abs().max() / r2.float().abs().max()).item()
                print(f"{name:<8} {variant:<14} layer {li:2d}: prefill(T=2) rel err {perr:.4f}", flush=True)
            err = (ref - out).abs().max().item() / ref.abs().max().item()
            # the caches must agree too
            k1, k2 = c1.layers[li].keys, c2.layers[li].keys
            kerr = (k1.float() - k2.float()).abs().max().item() / (k1.float().abs().max().item() + 1e-6)
            x, mask, pos_ids, pos, pe = last
            t_ref = graph_time_us(lambda: layer(x, attention_mask=mask, position_ids=pos_ids, past_key_values=c1, use_cache=True, cache_position=pos, position_embeddings=pe))
            t_fused = graph_time_us(lambda: fused(x, attention_mask=mask, position_ids=pos_ids, past_key_values=c2, use_cache=True, cache_position=pos, position_embeddings=pe))
            print(f"{name:<8} {variant:<14} layer {li:2d}: rel err out {err:.4f} cache {kerr:.4f} | step {t_ref:6.1f}us -> {t_fused:6.1f}us", flush=True)


if __name__ == "__main__":
    main()
