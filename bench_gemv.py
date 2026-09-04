"""Per-call time of the fused GEMV against what the model uses today."""
import sys, time, torch
sys.path.insert(0, ".")
from models.fused.kernels import FusedWeight, rms_gemv

dev = "cuda:0"
from models.fused.kernels import graph_time_us as t

M = 2
for name, K, N, epi in (("bb qkv 2048->4096", 2048, 4096, 0), ("bb gate|up 2048->2x6144 silu", 2048, 12288, 1), ("bb down 6144->2048 +res", 6144, 2048, 2),
                        ("dd gate|up 1024->2x8192 silu", 1024, 16384, 1), ("dd down 8192->1024 +res", 8192, 1024, 2), ("dd qkv 1024->1536", 1024, 1536, 0)):
    W = torch.randn(N, K, device=dev) * 0.02
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    nw = torch.ones(K, device=dev, dtype=torch.bfloat16)
    res = torch.randn(M, N // 2 if epi == 1 else N, device=dev, dtype=torch.bfloat16)
    Wb = W.to(torch.bfloat16)
    def ref(w_eff):
        xn = torch.nn.functional.rms_norm(x.float(), (K,), eps=1e-6).to(torch.bfloat16)
        y = xn.float() @ w_eff.float().T
        if epi == 1: g, u = y[:, :N // 2], y[:, N // 2:]; return torch.nn.functional.silu(g) * u
        if epi == 2: return y + res.float()
        return y
    base_bf16 = t(lambda: torch.nn.functional.linear(x, Wb))
    line = f"{name:<32} bf16 cublas {base_bf16:6.1f}us"
    for bits in (16, 8, 4):
        try:
            fw = FusedWeight(W, bits)
            out = rms_gemv(x, fw, nw, epilogue=epi, residual=res if epi == 2 else None)
            # reference with the same dequantised weights
            if bits == 16: w_eff = Wb
            elif bits == 8: w_eff = fw.w.float() * fw.s[:, None]
            else:
                q = torch.stack([fw.w & 15, fw.w >> 4], -1).reshape(N, K).float()
                w_eff = q * fw.s.float().repeat_interleave(128, 1) + fw.z.float().repeat_interleave(128, 1)
            err = (out.float() - ref(w_eff)).abs().max().item() / (ref(w_eff).abs().max().item() + 1e-6)
            us = t(lambda: rms_gemv(x, fw, nw, epilogue=epi, residual=res if epi == 2 else None))
            line += f" | fused{bits:>2} {us:6.1f}us err {err:.3f}"
        except Exception as ex:
            line += f" | fused{bits:>2} FAIL {type(ex).__name__}: {str(ex)[:60]}"
    if K % 128 == 0 and N >= 2048:
        try:
            from models.int4_linear import Int4Linear
            lin = torch.nn.Linear(K, N, bias=False).to(dev); lin.weight.data = Wb
            q4 = Int4Linear(lin)
            line += f" | tinygemm {t(lambda: q4(x)):6.1f}us"
        except Exception as ex:
            line += f" | tinygemm FAIL {str(ex)[:40]}"
    try:
        from models.fp8_linear import Fp8Linear as FP8Linear
        lin = torch.nn.Linear(K, N, bias=False).to(dev); lin.weight.data = Wb
        f8 = FP8Linear(lin)
        line += f" | scaled_mm {t(lambda: f8(x)):6.1f}us"
    except Exception as ex:
        line += f" | scaled_mm n/a"
    from models.fused.kernels import _BEST
    print(line, "| cfg", [v for k, v in _BEST.items() if k[0] == N and k[1] == K], flush=True)
