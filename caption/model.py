"""Captioner: backbone latents -> prefix tokens -> small causal LM -> prose.

The latents are (L, T, 2048): L backbone layers over T audio frames. A
learned softmax over layers picks the mix, a small transformer reads the
frames, and N learned queries cross-attend into them to produce N prefix
embeddings in the LM's input space (Perceiver-style resampler). The LM is
Qwen3-0.6B, fine-tuned end to end with a lower learning rate than the
projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

LM_NAME = "Qwen/Qwen3-0.6B"
LATENT_DIM = 2048
MAX_FRAMES = 128
PROMPT = "Describe the voice:\n"


@dataclass
class ProjectorConfig:
    kind: str = "pool"  # "pool": mean+std over time -> MLP; "attn": resampler
    num_layers_in: int = 4
    dim: int = 1024
    heads: int = 8
    encoder_layers: int = 2
    queries: int = 32
    dropout: float = 0.1


class Resampler(nn.Module):
    def __init__(self, cfg: ProjectorConfig, lm_dim: int):
        super().__init__()
        self.cfg = cfg
        self.layer_logits = nn.Parameter(torch.zeros(cfg.num_layers_in))
        # Per-dim standardisation with training-set statistics. The raw
        # states share a large mean vector across every clip; without removing
        # it, attention over the frames returns that constant plus a few
        # percent of voice, and the LM never sees the voice (prefix cosine
        # across clips was 1.0000 in the dry run). Set by set_input_stats.
        self.register_buffer("in_mean", torch.zeros(cfg.num_layers_in, 1, LATENT_DIM))
        self.register_buffer("in_std", torch.ones(cfg.num_layers_in, 1, LATENT_DIM))
        self.in_norm = nn.LayerNorm(LATENT_DIM)
        self.proj = nn.Linear(LATENT_DIM, cfg.dim)
        self.pos = nn.Parameter(torch.randn(MAX_FRAMES, cfg.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.dim,
            cfg.heads,
            cfg.dim * 4,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.encoder_layers, norm=nn.LayerNorm(cfg.dim)
        )
        self.queries = nn.Parameter(torch.randn(cfg.queries, cfg.dim) * 0.02)
        self.cross = nn.MultiheadAttention(
            cfg.dim, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.q_norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, lm_dim * 2),
            nn.GELU(),
            nn.Linear(lm_dim * 2, lm_dim),
        )
        # Prefix vectors are rms-normalised then scaled to the LM's token
        # embedding rms (set by the Captioner), with a learned gain.
        self.register_buffer("emb_rms", torch.tensor(1.0))
        self.gain = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def set_input_stats(self, latents: list[torch.Tensor]) -> None:
        """Per-layer, per-dim mean and std over a sample of (L, T, D) records."""
        x = torch.cat([r.float() for r in latents], dim=1)  # (L, sumT, D)
        self.in_mean.copy_(x.mean(1, keepdim=True))
        self.in_std.copy_(x.std(1, keepdim=True).clamp_min(1e-3))

    def forward(self, latents: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """latents (B, L, T, 2048) float; frame_mask (B, T) True where valid.
        Returns (B, Q, lm_dim)."""
        w = torch.softmax(self.layer_logits, dim=0).view(1, -1, 1, 1)
        z = (latents - self.in_mean) / self.in_std
        x = (self.in_norm(z) * w).sum(1)  # (B, T, 2048)
        x = self.proj(x) + self.pos[: x.shape[1]]
        pad = ~frame_mask
        x = self.encoder(x, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross(self.q_norm(q), x, x, key_padding_mask=pad)
        out = self.out(q + attended)
        out = out / out.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return out * (self.emb_rms * self.gain)


def pool_features(latents: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """(B, L, T, D) + (B, T) -> (B, L, 2D): masked mean and std over time."""
    m = frame_mask[:, None, :, None].float()
    n = m.sum(2).clamp_min(1.0)
    mean = (latents * m).sum(2) / n
    var = (((latents - mean.unsqueeze(2)) ** 2) * m).sum(2) / n
    return torch.cat([mean, var.clamp_min(0).sqrt()], dim=-1)


class PoolProjector(nn.Module):
    """Pooled features -> standardise with dataset stats -> MLP -> Q prefix
    vectors. The same features a linear probe reads sex, age and accent from,
    so there is no constant component for attention to amplify."""

    def __init__(self, cfg: ProjectorConfig, lm_dim: int):
        super().__init__()
        self.cfg = cfg
        feat = cfg.num_layers_in * 2 * LATENT_DIM
        self.register_buffer("in_mean", torch.zeros(cfg.num_layers_in, 2 * LATENT_DIM))
        self.register_buffer("in_std", torch.ones(cfg.num_layers_in, 2 * LATENT_DIM))
        self.net = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feat, cfg.dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim * 2, cfg.dim * 2),
            nn.GELU(),
            nn.Linear(cfg.dim * 2, cfg.queries * lm_dim),
        )
        self.lm_dim = lm_dim
        self.register_buffer("emb_rms", torch.tensor(1.0))
        self.gain = nn.Parameter(torch.tensor(1.0))
        # Kept so logs can print something; the pooled projector does not
        # weight layers explicitly, the MLP sees all of them.
        self.layer_logits = nn.Parameter(
            torch.zeros(cfg.num_layers_in), requires_grad=False
        )

    @torch.no_grad()
    def set_input_stats(self, latents: list[torch.Tensor]) -> None:
        feats = torch.stack(
            [
                pool_features(
                    r.float().unsqueeze(0),
                    torch.ones(1, r.shape[1], dtype=torch.bool, device=r.device),
                )[0]
                for r in latents
            ]
        )  # (N, L, 2D)
        self.in_mean.copy_(feats.mean(0))
        self.in_std.copy_(feats.std(0).clamp_min(1e-3))

    def forward(self, latents: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        z = (pool_features(latents, frame_mask) - self.in_mean) / self.in_std
        out = self.net(z.flatten(1)).view(-1, self.cfg.queries, self.lm_dim)
        out = out / out.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return out * (self.emb_rms * self.gain)


def build_attr_vocab(records: list[dict]) -> dict[str, list[str]]:
    vocab: dict[str, set[str]] = {}
    for r in records:
        for k, v in r["attrs"].items():
            vocab.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in sorted(vocab.items())}


class Captioner(nn.Module):
    def __init__(
        self,
        lm_name: str = LM_NAME,
        cfg: ProjectorConfig | None = None,
        tokenizer_name: str = LM_NAME,
        attr_vocab: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        # Auxiliary attribute classifiers on the prefix. Without them the
        # projector collapses to a constant soft prompt (prefix cosine across
        # clips went from 0.2 untrained to 0.99 after two epochs): carrying
        # caption style is the fastest way down early on, and once the LM has
        # learned the prior there is no gradient left to bring the voice back.
        self.attr_vocab = attr_vocab or {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name, dtype=torch.bfloat16)
        self.cfg = cfg or ProjectorConfig()
        cls = PoolProjector if self.cfg.kind == "pool" else Resampler
        self.resampler = cls(self.cfg, self.lm.config.hidden_size)
        with torch.no_grad():
            emb = self.lm.get_input_embeddings().weight.float()
            self.resampler.emb_rms.fill_(emb.pow(2).mean().sqrt().item())
        prompt_ids = self.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
        self.register_buffer("prompt_ids", torch.tensor(prompt_ids), persistent=False)
        lm_dim = self.lm.config.hidden_size
        self.aux_heads = nn.ModuleDict(
            {k: nn.Linear(lm_dim, len(v)) for k, v in self.attr_vocab.items()}
        )

    def aux_loss(
        self, prefix: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Mean CE over attributes, on the mean prefix vector. -100 = unstated."""
        if not self.aux_heads:
            return prefix.new_zeros(())
        # The prefix is scaled to the LM's embedding rms (0.03); a linear head
        # on that starts with near-zero logits and crawls. Normalise first.
        pooled = F.layer_norm(prefix.float().mean(1), prefix.shape[-1:])
        losses = []
        for k, head in self.aux_heads.items():
            t = targets[k]
            if (t != -100).any():
                losses.append(F.cross_entropy(head(pooled), t, ignore_index=-100))
        return torch.stack(losses).mean() if losses else prefix.new_zeros(())

    @property
    def device(self):
        return self.prompt_ids.device

    def _prefix(self, latents, frame_mask):
        embed = self.lm.get_input_embeddings()
        prefix = self.resampler(latents.float(), frame_mask).to(embed.weight.dtype)
        prompt = embed(self.prompt_ids).unsqueeze(0).expand(prefix.shape[0], -1, -1)
        return torch.cat([prefix, prompt], dim=1)

    def forward(self, latents, frame_mask, target_ids, target_mask, attr_targets=None):
        """Teacher-forced LM loss on the target tokens (+ aux attribute loss
        when attr_targets is given): returns (lm_loss, aux_loss)."""
        prefix = self._prefix(latents, frame_mask)
        aux = (
            self.aux_loss(prefix[:, : self.cfg.queries], attr_targets)
            if attr_targets is not None
            else prefix.new_zeros(())
        )
        embed = self.lm.get_input_embeddings()
        inputs = torch.cat([prefix, embed(target_ids)], dim=1)
        attn = torch.cat(
            [
                torch.ones(prefix.shape[:2], dtype=torch.long, device=inputs.device),
                target_mask,
            ],
            dim=1,
        )
        labels = torch.cat(
            [
                torch.full(
                    prefix.shape[:2], -100, dtype=torch.long, device=inputs.device
                ),
                target_ids.masked_fill(target_mask == 0, -100),
            ],
            dim=1,
        )
        out = self.lm(inputs_embeds=inputs, attention_mask=attn)
        logits = out.logits[:, :-1].float()
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return lm_loss, aux

    @torch.no_grad()
    def generate(
        self, latents, frame_mask, max_new_tokens: int = 96, **kw
    ) -> list[str]:
        prefix = self._prefix(latents, frame_mask)
        attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        out = self.lm.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kw,
        )
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in out
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(path / "lm")
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "resampler": self.resampler.state_dict(),
                "tokenizer": self.tokenizer.name_or_path,
                "attr_vocab": self.attr_vocab,
                "aux_heads": self.aux_heads.state_dict(),
            },
            path / "resampler.pt",
        )

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> Captioner:
        state = torch.load(path / "resampler.pt", map_location="cpu")
        model = cls(
            str(path / "lm"),
            ProjectorConfig(**state["cfg"]),
            tokenizer_name=state.get("tokenizer", LM_NAME),
            attr_vocab=state.get("attr_vocab"),
        )
        model.resampler.load_state_dict(state["resampler"])
        if "aux_heads" in state:
            model.aux_heads.load_state_dict(state["aux_heads"])
        return model.to(device).eval()


def attr_targets(
    items: list[dict], vocab: dict[str, list[str]]
) -> dict[str, torch.Tensor]:
    out = {}
    for k, values in vocab.items():
        idx = {v: i for i, v in enumerate(values)}
        out[k] = torch.tensor(
            [idx.get(it.get("attrs", {}).get(k), -100) for it in items]
        )
    return out


def collate(
    items: list[dict], tokenizer, max_frames: int = MAX_FRAMES, max_tokens: int = 128
):
    """Pad latents over time and tokenise prose (with EOS) for a batch."""
    L = items[0]["latents"].shape[0]
    T = min(max(it["latents"].shape[1] for it in items), max_frames)
    B = len(items)
    latents = torch.zeros(B, L, T, LATENT_DIM, dtype=torch.bfloat16)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    for b, it in enumerate(items):
        x = it["latents"][:, :T]
        latents[b, :, : x.shape[1]] = x
        frame_mask[b, : x.shape[1]] = True
    enc = tokenizer(
        [it["prose"] + tokenizer.eos_token for it in items],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    return latents, frame_mask, enc["input_ids"], enc["attention_mask"]
