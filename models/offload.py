"""Keep cold parameters in host memory instead of VRAM.

Embeddings are lookup tables, not matmuls: a forward pass gathers a handful of
rows. The text-side tables total ~1.7 GB but are touched *once per request*, and
a gather of ~500 rows moves about 2 MB. Holding them in VRAM buys nothing on a
memory-constrained card.

The depth decoder's embedding is the opposite case -- hit once per codebook,
sixteen times per audio frame, ~200 lookups/second. Offloading that would pay a
host round trip per lookup. It stays resident, and is deliberately excluded
below.

Offload is preferable to quantising these tables: it costs no accuracy at all,
whereas embeddings are exactly what mixed-precision schemes keep at higher
precision.

PyTorch does not move tensors across devices implicitly, so each offloaded
module is wrapped to shuttle its inputs and outputs.
"""

from __future__ import annotations

import torch
from torch import nn

# Text-side tables only. Names are checked against the module tree at call time,
# so a rename surfaces as "not found" rather than silently offloading nothing.
TEXT_EMBEDDING_PATHS = (
    "embed_text_tokens",
    "text_encoder.embed_tokens",
    "text_encoder.model.embed_tokens",
)


class CpuOffloaded(nn.Module):
    """Run a module on the host, moving inputs in and results back."""

    def __init__(self, inner: nn.Module, device: torch.device | str) -> None:
        super().__init__()
        self.inner = inner.to("cpu")
        self.target_device = torch.device(device)

    def forward(self, *args, **kwargs):
        moved_args = [
            a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args
        ]
        moved_kwargs = {
            k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }
        out = self.inner(*moved_args, **moved_kwargs)
        if isinstance(out, torch.Tensor):
            return out.to(self.target_device)
        if isinstance(out, tuple):
            return tuple(
                o.to(self.target_device) if isinstance(o, torch.Tensor) else o
                for o in out
            )
        return out

    def _apply(self, fn, recurse: bool = True):
        """Pin to host memory: ignore .to()/.cuda() from enclosing modules.

        Without this, a later ``model.to(device)`` walks into the wrapper and
        drags the table onto the GPU -- transiently spending the very VRAM the
        offload exists to save, which is enough to OOM a small card even though
        the steady state would fit.
        """
        return self

    def extra_repr(self) -> str:
        return f"offloaded_to=cpu, returns_to={self.target_device}"


def _resolve(root: nn.Module, path: str) -> tuple[nn.Module, str] | None:
    """Return (parent, attribute) for a dotted path, or None if absent."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None
        parent = getattr(parent, part)
    return (parent, parts[-1]) if hasattr(parent, parts[-1]) else None


def offload_text_embeddings(
    model: nn.Module, device: torch.device | str
) -> dict[str, object]:
    """Move text-side embedding tables to host memory.

    Returns which paths were offloaded and how much VRAM that frees, so a
    missed rename is visible rather than silently doing nothing.
    """
    offloaded: list[str] = []
    freed_bytes = 0

    for path in TEXT_EMBEDDING_PATHS:
        resolved = _resolve(model, path)
        if resolved is None:
            continue
        parent, attr = resolved
        module = getattr(parent, attr)
        if isinstance(module, CpuOffloaded) or not isinstance(module, nn.Module):
            continue
        params = sum(p.numel() * p.element_size() for p in module.parameters())
        if params == 0:
            continue
        setattr(parent, attr, CpuOffloaded(module, device))
        offloaded.append(path)
        freed_bytes += params

    return {
        "offloaded": offloaded,
        "not_found": [p for p in TEXT_EMBEDDING_PATHS if p not in offloaded],
        "vram_bytes_freed": freed_bytes,
    }
