"""Where does a frame's time go? backbone step vs 15 depth-decoder steps vs codec.

Builds the engine the way kokoro_compat_server does, wraps the three stage
calls with synchronised timers, renders a sentence with the default voice.

    python bench_frame_split.py /path/to/breeze [--fp8 depth --int4 backbone ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

import kokoro_compat_server as srv  # noqa: E402


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--voices-dir", type=Path, default=Path("voices"))
    p.add_argument("--fp8", default="off")
    p.add_argument("--int4", default="off")
    p.add_argument("--device", default=None)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--fast-profile", type=Path, default=Path("configs/fast_service.json"))
    p.add_argument("--fused", action="store_true")
    p.add_argument("--fused-attn-bits", type=int, default=16)
    p.add_argument("--wall", action="store_true", help="no stage timers (they sync); wall clock only")
    p.add_argument("--out", type=Path, help="save the timed render as wav")
    p.add_argument("--next-text", default="", help="exercise the lookahead stop rule")
    args = p.parse_args()

    engine = srv.BreezeEngine(
        args.model, cfg_scale=args.cfg_scale, seed=42, fast=True, fast_stages="decode",
        fast_profile=args.fast_profile, fp8=args.fp8, int4=args.int4, device=args.device,
        fused=args.fused, fused_attn_bits=args.fused_attn_bits,
    )
    rt = engine.runtime
    acc = {"backbone": 0.0, "depth": 0.0, "codec": 0.0}
    calls = {"backbone": 0, "depth": 0, "codec": 0}

    def timed(name, fn):
        def w(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            acc[name] += time.perf_counter() - t
            calls[name] += 1
            return out
        return w

    if not args.wall:
        rt._backbone_graph.run = timed("backbone", rt._backbone_graph.run)
        rt._depth_decoder_graph.run = timed("depth", rt._depth_decoder_graph.run)
        rt._decode_codec_frames = timed("codec", rt._decode_codec_frames)

    voice = srv.VoiceLibrary(args.voices_dir, "default").get("default")
    text = "It was not an unfriendly silence, Mira had learned over three seasons, but the silence of people who had run out of things to say."
    engine.synthesize(text, voice)  # warm
    for k in acc:
        acc[k] = 0.0
        calls[k] = 0
    t = time.perf_counter()
    audio, turns = engine.synthesize_stateless(text, voice, [], next_text=args.next_text)
    wall = time.perf_counter() - t
    sec = len(audio) / engine.sample_rate
    frames = calls["depth"] or int(round(sec * 12.5))
    mode = "fused" if args.fused else "stock"
    print(f"{mode}: {sec:.1f}s audio, {frames} frames, wall {wall:.2f}s = {sec / wall:.2f}x realtime, {wall / frames * 1000:.1f} ms/frame, cond {sum(len(t.codes) for t in turns)} codes")
    if args.out:
        import soundfile as sf

        sf.write(args.out, audio, engine.sample_rate)
    if args.wall:
        return
    for k in ("backbone", "depth", "codec"):
        print(f"  {k:<9} {acc[k] / frames * 1000:6.1f} ms/frame  ({calls[k]} calls, {acc[k] / max(calls[k], 1) * 1000:.2f} ms each)")
    other = wall - sum(acc.values())
    print(f"  other     {other / frames * 1000:6.1f} ms/frame  (python, sampling, syncs)")


if __name__ == "__main__":
    main()
