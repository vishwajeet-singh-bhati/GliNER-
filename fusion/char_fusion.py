"""Cross-stream gated fusion for GLiNER (professor's suggestion).

Semantic stream  : the word representation GLiNER already has (mDeBERTa -> proj -> BiLSTM), shape (B, W, D).
Structural stream: a char-CNN over the RAW CHARACTERS of the same words (same word boundaries as the
                   semantic stream, NOT its token embeddings) + explicit shape statistics (digit density...).
Fusion           : g = sigmoid(Linear([sem ; struct])),  fused = g * sem + (1 - g) * struct
Insertion point  : directly before span_rep_layer, i.e. before the span-vs-label dot product.

Nothing inside the quantized encoder changes, so the E8 QAT recipe is unaffected.

Usage:
    model = GLiNER.from_pretrained(...)
    attach_char_fusion(model)            # adds modules + patches collator/forward
    # then load a fusion checkpoint with the normal state-dict loader, or train.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHAR_IDS = 258           # 0 pad, 1..256 = latin-1 code points, 257 = any other char
N_STATS = 6


def char_to_id(ch: str) -> int:
    o = ord(ch)
    return o + 1 if o < 256 else 257


def build_char_ids(words_batch, max_chars: int = 32):
    """words_batch: list (B) of list (W_b) of str -> LongTensor (B, W, max_chars), 0 = pad.
    Long words keep their first and last chars (head/tail carry prefix/suffix formatting)."""
    W = max((len(w) for w in words_batch), default=1) or 1
    out = torch.zeros(len(words_batch), W, max_chars, dtype=torch.long)
    for b, words in enumerate(words_batch):
        for i, w in enumerate(words):
            if len(w) > max_chars:
                h = max_chars // 2
                w = w[:h] + w[-(max_chars - h):]
            if w:
                out[b, i, : len(w)] = torch.tensor([char_to_id(c) for c in w])
    return out


class CharCNN(nn.Module):
    def __init__(self, out_dim: int, emb_dim: int = 32, channels: int = 64, kernels=(2, 3, 4)):
        super().__init__()
        self.emb = nn.Embedding(N_CHAR_IDS, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k - 1) for k in kernels])
        self.proj = nn.Linear(channels * len(kernels) + N_STATS, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @staticmethod
    def stats(ids: torch.Tensor) -> torch.Tensor:
        """Explicit formatting features per word from the raw chars: digit / upper / lower / punct
        fractions, log length, has-any-digit.  (ids: (N, C) with the latin-1 id = ord + 1)."""
        valid = ids > 0
        n = valid.sum(-1).clamp(min=1).float()
        o = (ids - 1).clamp(min=0)
        digit = valid & (o >= 48) & (o <= 57)
        upper = valid & (o >= 65) & (o <= 90)
        lower = valid & (o >= 97) & (o <= 122)
        punct = valid & ~digit & ~upper & ~lower
        f = lambda m: m.sum(-1).float() / n
        return torch.stack([f(digit), f(upper), f(lower), f(punct),
                            torch.log1p(valid.sum(-1).float()) / 4.0, digit.any(-1).float()], dim=-1)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:          # (B, W, C) -> (B, W, D)
        B, W, C = char_ids.shape
        flat = char_ids.reshape(B * W, C)
        x = self.emb(flat).transpose(1, 2)                                # (N, E, C)
        feats = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        h = torch.cat(feats + [self.stats(flat).to(x.dtype)], dim=-1)
        h = self.norm(self.proj(h)).view(B, W, -1)
        return h * (char_ids.sum(-1, keepdim=True) > 0)                   # padded words -> 0


class GatedFusion(nn.Module):
    def __init__(self, dim: int, gate_bias: float = 4.0):
        super().__init__()
        self.char_cnn = CharCNN(dim)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, gate_bias)    # g ~ 0.98 -> starts as the pretrained semantic model
        self.last_gate_mean = None

    def forward(self, sem: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        W = sem.size(1)
        if char_ids.size(1) < W:
            char_ids = F.pad(char_ids, (0, 0, 0, W - char_ids.size(1)))
        char_ids = char_ids[:, :W].to(sem.device)
        struct = self.char_cnn(char_ids).to(sem.dtype)
        g = torch.sigmoid(self.gate(torch.cat([sem, struct], dim=-1)))
        self.last_gate_mean = g.detach().mean()
        return g * sem + (1.0 - g) * struct


class FusedSpanRep(nn.Module):
    """Drop-in for span_rep_layer: fuses word reps, then calls the original layer."""

    def __init__(self, span_rep_layer: nn.Module, dim: int, gate_bias: float = 4.0):
        super().__init__()
        self.span_rep_layer = span_rep_layer
        self.fusion = GatedFusion(dim, gate_bias)
        self.char_ids = None                       # set per forward by the patched model.forward

    def forward(self, words_embedding, span_idx, *a, **kw):
        if self.char_ids is not None:
            words_embedding = self.fusion(words_embedding, self.char_ids)
        return self.span_rep_layer(words_embedding, span_idx, *a, **kw)


def _wrap_collator(base_cls, max_chars):
    class CharFusionCollator(base_cls):
        def __call__(self, input_x, *a, **kw):
            batch = super().__call__(input_x, *a, **kw)
            batch["char_ids"] = build_char_ids([list(x["tokenized_text"]) for x in input_x], max_chars)
            return batch
    CharFusionCollator.__name__ = "CharFusionCollator"
    return CharFusionCollator


def attach_char_fusion(gliner_model, gate_bias: float = 4.0, max_chars: int = 32):
    """Idempotent. Call BEFORE loading a fusion checkpoint's state dict and BEFORE training."""
    core = gliner_model.model
    if isinstance(core.span_rep_layer, FusedSpanRep):
        return gliner_model
    dim = core.config.hidden_size
    wrapper = FusedSpanRep(core.span_rep_layer, dim, gate_bias)
    core.span_rep_layer = wrapper
    orig_forward = core.forward

    def forward(*a, char_ids=None, **kw):
        wrapper.char_ids = char_ids
        try:
            return orig_forward(*a, **kw)
        finally:
            wrapper.char_ids = None
    core.forward = forward
    gliner_model.data_collator_class = _wrap_collator(gliner_model.data_collator_class, max_chars)
    return gliner_model


def is_fusion_state_dict(sd) -> bool:
    return any("span_rep_layer.fusion." in k for k in sd)


# ----------------------------------------------------------------------------- selftest
def selftest():
    torch.manual_seed(0)
    ids = build_char_ids([["John", "4111-1111", "a" * 80], ["x"]], 32)
    assert ids.shape == (2, 3, 32) and ids[1, 1].sum() == 0
    st = CharCNN.stats(ids[0])
    assert abs(st[1, 0].item() - 8 / 9) < 1e-5 and st[0, 5] == 0 and st[1, 5] == 1   # digit density
    D = 16
    fus = GatedFusion(D)
    sem = torch.randn(2, 5, D)                       # W=5 > chars W=3 -> padded
    out = fus(sem, ids)
    assert out.shape == sem.shape
    assert fus.last_gate_mean > 0.97
    assert torch.allclose(out[1, 4], sem[1, 4] * fus.last_gate_mean, atol=0.05)   # padded word ~ semantic only
    (out.sum()).backward()
    assert fus.char_cnn.emb.weight.grad is not None and fus.gate.weight.grad is not None

    class Cfg: hidden_size = D; max_width = 12
    class Ident(nn.Module):
        def forward(s, x, span_idx): return x
    class Core(nn.Module):
        def __init__(s):
            super().__init__(); s.config = Cfg(); s.span_rep_layer = Ident(); s.seen = None
        def forward(s, x, span_idx=None, **kw):
            s.seen = s.span_rep_layer(x, span_idx); return s.seen
    class G: pass
    g = G(); g.model = Core(); g.data_collator_class = object
    attach_char_fusion(g); attach_char_fusion(g)     # idempotent
    x = torch.randn(2, 5, D)
    g.model(x, char_ids=ids); y = g.model.seen
    g.model(x); z = g.model.seen                       # no char ids -> unchanged
    assert not torch.equal(x, y) and torch.equal(x, z)
    sd = g.model.state_dict()
    assert is_fusion_state_dict(sd)
    print(f"selftest OK  params(fusion)={sum(p.numel() for p in g.model.span_rep_layer.fusion.parameters()):,}  gate~{fus.last_gate_mean:.3f}")


if __name__ == "__main__":
    selftest()
