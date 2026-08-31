"""Correctness + speed check for Fp8Linear before wiring it into the model."""

import torch
from torch import nn

from models.fp8_linear import Fp8Linear, quantize_module_fp8

DEV = "cuda"


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(0), torch.__version__)

    print("\n--- numerics (vs bf16 reference) ---")
    for k, n in ((1024, 8192), (8192, 1024)):
        lin = nn.Linear(k, n, bias=False, device=DEV, dtype=torch.bfloat16)
        q = Fp8Linear(lin).to(DEV)
        for m in (1, 2):
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            ref = lin(x)
            got = q(x)
            print(
                f"  {k}->{n} M={m}: rel_err {rel_err(got.float(), ref.float()):.4f}  "
                f"dtype {got.dtype} shape {tuple(got.shape)}"
            )

    print("\n--- 3D input (batch, seq, hidden) ---")
    lin = nn.Linear(1024, 8192, bias=False, device=DEV, dtype=torch.bfloat16)
    q = Fp8Linear(lin).to(DEV)
    x = torch.randn(2, 3, 1024, device=DEV, dtype=torch.bfloat16)
    print(f"  out shape {tuple(q(x).shape)} rel_err {rel_err(q(x).float(), lin(x).float()):.4f}")

    print("\n--- torch.compile(fullgraph=True) ---")
    try:
        compiled = torch.compile(q, mode="default", fullgraph=True)
        x = torch.randn(2, 1024, device=DEV, dtype=torch.bfloat16)
        out = compiled(x)
        print(f"  OK  rel_err {rel_err(out.float(), lin(x).float()):.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    print("\n--- CUDA graph capture ---")
    try:
        static_x = torch.randn(2, 1024, device=DEV, dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                q(static_x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = q(static_x)
        graph.replay()
        torch.cuda.synchronize()
        print(f"  OK  captured+replayed, out shape {tuple(static_out.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    print("\n--- swap helper on a toy MLP ---")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(1024, 8192, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(1024, 8192, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(8192, 1024, bias=False, dtype=torch.bfloat16)
            self.q_proj = nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16)

    mlp = MLP().to(DEV)
    stats = quantize_module_fp8(mlp)
    print(f"  {stats}")
    print(f"  gate={type(mlp.gate_proj).__name__} q_proj={type(mlp.q_proj).__name__}")


if __name__ == "__main__":
    main()
