"""Fused decode kernels for batch-2 CFG decoding.

The frame loop runs ~8000 kernels a frame; the GEMMs take ~11 µs each on a
5090 where their weight traffic is ~1 µs. These kernels fold the norm, the
dequantisation and the epilogue into the weight-streaming pass, so a layer
step is a handful of launches instead of thirty.

rms_gemv: out = epilogue( rmsnorm(x) @ W^T )
  W is bf16 (N, K); int4 (N, K/2) packed with per-group scale/min; or fp8
  e4m3 (N, K) with a per-row scale. M (rows of x) is small; rows are padded
  to 16 so tensor cores do the (tiny) arithmetic.
  epilogue 0: none; 1: silu(gate) * up over a [gate | up] weight, output N/2;
  2: add residual.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP = 128
M_PAD = 16


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


def _configs():
    out = []
    for bn in (16, 32, 64):
        for sk in (1, 2, 4, 8, 16):
            for nw in (2, 4):
                for st in (2, 4):
                    out.append(triton.Config({"BLOCK_N": bn, "SPLIT_K": sk}, num_warps=nw, num_stages=st))
    return out


@triton.jit
def rms_gemv_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, nw_ptr, part_ptr,
    M, N, K, eps,
    stride_x,
    WBITS: tl.constexpr, NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_K: tl.constexpr,
):
    """Partial products over a K slice into part[split, M, N] (fp32; the fp8
    row scale is applied in the epilogue). Every program recomputes the rms
    of x: a K-long read from L2, cheaper than a launch."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, 16)
    rmask = rows < M
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    k_per = K // SPLIT_K
    k_lo = pid_k * k_per
    NG: tl.constexpr = BLOCK_K // GROUP_K

    inv = tl.zeros([16], dtype=tl.float32)
    if NORM:
        ss = tl.zeros([16], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            xk = tl.load(x_ptr + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            ss += tl.sum(xk * xk, 1)
        inv = 1.0 / tl.sqrt(ss / K + eps)

    acc = tl.zeros([16, BLOCK_N], dtype=tl.float32)
    for k0 in range(k_lo, k_lo + k_per, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        xk = tl.load(x_ptr + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        if NORM:
            nw = tl.load(nw_ptr + ks).to(tl.float32)
            xk = xk * inv[:, None] * nw[None, :]
        xb = xk.to(tl.bfloat16)
        if WBITS == 16:
            w = tl.load(w_ptr + n[:, None] * K + ks[None, :], mask=nmask[:, None], other=0.0)
        elif WBITS == 8:
            w = tl.load(w_ptr + n[:, None] * K + ks[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        else:
            # int4: byte j of row n holds k=2j (low nibble) and k=2j+1 (high).
            kh = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            packed = tl.load(w_ptr + n[:, None] * (K // 2) + kh[None, :], mask=nmask[:, None], other=0)
            lo = (packed & 15).to(tl.float32)
            hi = (packed >> 4).to(tl.float32)
            q = tl.reshape(tl.join(lo, hi), (BLOCK_N, NG, GROUP_K))
            gs = k0 // GROUP_K + tl.arange(0, NG)
            sc = tl.load(s_ptr + n[:, None] * (K // GROUP_K) + gs[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            mn = tl.load(z_ptr + n[:, None] * (K // GROUP_K) + gs[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            w = tl.reshape(q * sc[:, :, None] + mn[:, :, None], (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
        acc += tl.dot(xb, tl.trans(w))
    tl.store(part_ptr + (pid_k * M + rows[:, None]) * N + n[None, :], acc, mask=rmask[:, None] & nmask[None, :])


@triton.jit
def gemv_epilogue_kernel(
    part_ptr, s_ptr, res_ptr, out_ptr,
    M, N, SPLIT_K, stride_res, stride_out,
    WBITS: tl.constexpr, EPI: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    n = pid * BLOCK + tl.arange(0, BLOCK)
    if EPI == 1:
        half = N // 2
        nmask = n < half
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(SPLIT_K):
            acc += tl.load(part_ptr + (k * M + row) * N + n, mask=nmask, other=0.0)
            acc2 += tl.load(part_ptr + (k * M + row) * N + n + half, mask=nmask, other=0.0)
        if WBITS == 8:
            acc = acc * tl.load(s_ptr + n, mask=nmask, other=0.0)
            acc2 = acc2 * tl.load(s_ptr + n + half, mask=nmask, other=0.0)
        out = acc * tl.sigmoid(acc) * acc2
    else:
        nmask = n < N
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(SPLIT_K):
            acc += tl.load(part_ptr + (k * M + row) * N + n, mask=nmask, other=0.0)
        if WBITS == 8:
            acc = acc * tl.load(s_ptr + n, mask=nmask, other=0.0)
        if EPI == 2:
            acc += tl.load(res_ptr + row * stride_res + n, mask=nmask, other=0.0).to(tl.float32)
        out = acc
    tl.store(out_ptr + row * stride_out + n, out.to(out_ptr.dtype.element_ty), mask=nmask)


def pack_int4(weight: torch.Tensor, group: int = GROUP):
    """(N, K) float -> packed (N, K/2) uint8, scale (N, K/group) bf16, min (N, K/group) bf16.
    w ~= q * scale + min, q in 0..15. Even k in the low nibble."""
    N, K = weight.shape
    g = weight.float().reshape(N, K // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((g - lo) / scale).round().clamp(0, 15).to(torch.uint8).reshape(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1).to(torch.bfloat16).contiguous(), lo.squeeze(-1).to(torch.bfloat16).contiguous()


def pack_fp8(weight: torch.Tensor):
    """(N, K) float -> fp8 e4m3 (N, K) with a per-row fp32 scale."""
    amax = weight.float().abs().amax(-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 448.0
    q = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.squeeze(-1).float().contiguous()


class FusedWeight:
    """A weight in one of the three formats, ready for rms_gemv."""

    def __init__(self, weight: torch.Tensor, bits: int = 16):
        self.bits = bits
        self.N, self.K = weight.shape
        if bits == 16:
            self.w = weight.to(torch.bfloat16).contiguous()
            self.s = self.z = None
        elif bits == 8:
            self.w, self.s = pack_fp8(weight)
            self.z = None
        elif bits == 4:
            self.w, self.s, self.z = pack_int4(weight)
        else:
            raise ValueError(bits)


_PART: dict[tuple, torch.Tensor] = {}
_BEST: dict[tuple, tuple] = {}
# (BLOCK_N, SPLIT_K, num_warps, num_stages) candidates; a shape is timed once.
_CANDIDATES = [
    (bn, sk, bk, nw, st)
    for bn in (16, 32, 64)
    for sk in (1, 2, 4, 8, 16, 32, 64)
    for bk in (128, 256, 512)
    for nw in (2, 4)
    for st in (2, 3)
]


def _partial_buffer(N: int, device) -> torch.Tensor:
    key = (N, str(device))
    buf = _PART.get(key)
    if buf is None:
        buf = torch.empty(64 * 16, N, device=device, dtype=torch.float32)  # up to SPLIT_K=64, M<=16
        _PART[key] = buf
    return buf


def _launch(cfg, x, fw, norm_weight, eps, epilogue, residual, out, part):
    bn, sk, bk, nw, st = cfg
    M, K = x.shape
    N = fw.N
    n_out = N // 2 if epilogue == 1 else N
    dummy = out
    rms_gemv_kernel[(triton.cdiv(N, bn), sk)](
        x, fw.w, fw.s if fw.s is not None else dummy, fw.z if fw.z is not None else dummy,
        norm_weight if norm_weight is not None else dummy, part,
        M, N, K, eps, x.stride(0),
        WBITS=fw.bits, NORM=norm_weight is not None,
        BLOCK_N=bn, SPLIT_K=sk, BLOCK_K=bk, GROUP_K=GROUP, num_warps=nw, num_stages=st,
    )
    gemv_epilogue_kernel[(triton.cdiv(n_out, 1024), M)](
        part, fw.s if fw.s is not None else dummy, residual if residual is not None else dummy, out,
        M, N, sk, residual.stride(0) if residual is not None else 0, out.stride(0),
        WBITS=fw.bits, EPI=epilogue, BLOCK=1024, num_warps=4,
    )


def graph_time_us(fn, iters: int = 20) -> float:
    """GPU time per call of fn, measured as a CUDA graph replay: launch
    overhead is not what the frame loop pays, graphs are."""
    fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
    torch.cuda.current_stream().wait_stream(st)
    g.replay()
    torch.cuda.synchronize()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(5):
        g.replay()
    e_ev.record()
    torch.cuda.synchronize()
    return s_ev.elapsed_time(e_ev) / (5 * iters) * 1000


def _tune(key, x, fw, norm_weight, eps, epilogue, residual, out, part) -> tuple:
    best, best_t = None, float("inf")
    K = x.shape[1]
    for cfg in _CANDIDATES:
        bn, sk, bk, nw, st = cfg
        if (K // sk) % bk:
            continue
        try:
            t = graph_time_us(lambda: _launch(cfg, x, fw, norm_weight, eps, epilogue, residual, out, part))
        except Exception:
            continue
        if t < best_t:
            best, best_t = cfg, t
    if best is None:
        raise RuntimeError(f"no working config for {key}")
    _BEST[key] = best
    return best


def rms_gemv(x, fw: FusedWeight, norm_weight=None, eps: float = 1e-6, epilogue: int = 0, residual=None, out=None):
    M, K = x.shape
    assert K == fw.K and M <= M_PAD
    N = fw.N
    n_out = N // 2 if epilogue == 1 else N
    if out is None:
        out = torch.empty(M, n_out, device=x.device, dtype=torch.bfloat16 if residual is None else residual.dtype)
    part = _partial_buffer(N, x.device)
    key = (N, K, fw.bits, epilogue, norm_weight is not None, str(x.device))
    cfg = _BEST.get(key)
    if cfg is None:
        cfg = _tune(key, x, fw, norm_weight, eps, epilogue, residual, out, part)
    _launch(cfg, x, fw, norm_weight, eps, epilogue, residual, out, part)
    return out


# ---- attention for one decode step ------------------------------------------

@triton.jit
def _rms_head(v, w, eps, D: tl.constexpr):
    var = tl.sum(v * v, 0) / D
    return v * (1.0 / tl.sqrt(var + eps)) * w


@triton.jit
def _rope(v, cos, sin, D: tl.constexpr):
    # rotate_half: x1 = v[:D/2], x2 = v[D/2:]; out = v*cos + cat(-x2, x1)*sin
    half: tl.constexpr = D // 2
    x1 = tl.reshape(v, (2, half))
    i = tl.arange(0, 2)
    swap = tl.sum(tl.where(i[:, None] == 1, x1, 0.0), 0)  # x2
    first = tl.sum(tl.where(i[:, None] == 0, x1, 0.0), 0)  # x1
    # join adds a trailing dim: (half, 2); transpose so the reshape gives
    # [-x2..., x1...] rather than an interleave.
    rot = tl.reshape(tl.trans(tl.join(-swap, first)), (D,))
    return v * cos + rot * sin


@triton.jit
def qkv_rope_cache_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
    stride_qkv, stride_cs, eps,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, S: tl.constexpr, QK_NORM: tl.constexpr,
):
    """One program per (batch row, head). q heads: optional per-head rmsnorm,
    rope, written to q_out (B, HQ, D) fp32. kv heads: k normed+roped and v
    written into the static cache at cache position pos."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    cos = tl.load(cos_ptr + b * stride_cs + d).to(tl.float32)
    sin = tl.load(sin_ptr + b * stride_cs + d).to(tl.float32)
    if h < HQ:
        q = tl.load(qkv_ptr + b * stride_qkv + h * D + d).to(tl.float32)
        if QK_NORM:
            q = _rms_head(q, tl.load(qn_ptr + d).to(tl.float32), eps, D)
        q = _rope(q, cos, sin, D)
        tl.store(q_out_ptr + (b * HQ + h) * D + d, q)
    else:
        j = h - HQ
        pos = tl.load(pos_ptr)
        k = tl.load(qkv_ptr + b * stride_qkv + (HQ + j) * D + d).to(tl.float32)
        if QK_NORM:
            k = _rms_head(k, tl.load(kn_ptr + d).to(tl.float32), eps, D)
        k = _rope(k, cos, sin, D)
        v = tl.load(qkv_ptr + b * stride_qkv + (HQ + HKV + j) * D + d)
        base = ((b * HKV + j) * S + pos) * D
        tl.store(k_cache_ptr + base + d, k.to(k_cache_ptr.dtype.element_ty))
        tl.store(v_cache_ptr + base + d, v.to(v_cache_ptr.dtype.element_ty))


@triton.jit
def attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, mask_ptr, out_ptr,
    stride_mask, stride_out, scale,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """One program per (batch row, q head): softmax(q k^T * scale + mask) v over
    the whole static cache. mask is additive (B, S), float."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    j = h // (HQ // HKV)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (b * HQ + h) * D + d) * scale
    m = tl.full([1], -1e30, dtype=tl.float32)
    l = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    kv_base = (b * HKV + j) * S * D
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        k = tl.load(k_cache_ptr + kv_base + ss[:, None] * D + d[None, :]).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) + tl.load(mask_ptr + b * stride_mask + ss).to(tl.float32)
        m_new = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(v_cache_ptr + kv_base + ss[:, None] * D + d[None, :]).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l = l * alpha + tl.sum(p, 0)
        m = m_new
    out = acc / l
    tl.store(out_ptr + b * stride_out + h * D + d, out.to(out_ptr.dtype.element_ty))


def attention_step(qkv, cos, sin, cache_position, keys, values, mask, q_norm_w=None, k_norm_w=None, eps=1e-6, out=None, q_buf=None):
    """qkv (B, (HQ+2HKV)*D) bf16 -> attention output (B, HQ*D) bf16, with the
    new k/v written into the static cache (B, HKV, S, D)."""
    B = qkv.shape[0]
    _, HKV, S, D = keys.shape
    HQ = (qkv.shape[1] - 2 * HKV * D) // D
    if q_buf is None:
        q_buf = torch.empty(B, HQ, D, device=qkv.device, dtype=torch.float32)
    if out is None:
        out = torch.empty(B, HQ * D, device=qkv.device, dtype=torch.bfloat16)
    cos2 = cos.reshape(B, -1)
    sin2 = sin.reshape(B, -1)
    dummy = out
    qkv_rope_cache_kernel[(B, HQ + HKV)](
        qkv, q_norm_w if q_norm_w is not None else dummy, k_norm_w if k_norm_w is not None else dummy,
        cos2, sin2, cache_position, q_buf, keys, values,
        qkv.stride(0), cos2.stride(0), eps,
        HQ=HQ, HKV=HKV, D=D, S=S, QK_NORM=q_norm_w is not None, num_warps=1,
    )
    mask2 = mask.reshape(B, -1)
    attn_decode_kernel[(B, HQ)](
        q_buf, keys, values, mask2, out, mask2.stride(0), out.stride(0), D ** -0.5,
        HQ=HQ, HKV=HKV, D=D, S=S, BLOCK_S=64, num_warps=4,
    )
    return out


# ---- small glue -------------------------------------------------------------

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, K, eps, stride_x, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ks = tl.arange(0, BLOCK)
    mask = ks < K
    x = tl.load(x_ptr + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / K + eps)
    w = tl.load(w_ptr + ks, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + ks, (x * inv * w).to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x, w, eps=1e-6, out=None):
    M, K = x.shape
    if out is None:
        out = torch.empty_like(x)
    rmsnorm_kernel[(M,)](x, w, out, K, eps, x.stride(0), out.stride(0), BLOCK=triton.next_power_of_2(K), num_warps=8)
    return out


@triton.jit
def silu_mul_kernel(g_ptr, u_ptr, out_ptr, N, stride_g, stride_u, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    g = tl.load(g_ptr + row * stride_g + n, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * stride_u + n, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + n, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(g, u, out=None):
    M, N = g.shape
    if out is None:
        out = torch.empty_like(g)
    silu_mul_kernel[(triton.cdiv(N, 1024), M)](g, u, out, N, g.stride(0), u.stride(0), out.stride(0), BLOCK=1024, num_warps=4)
    return out
