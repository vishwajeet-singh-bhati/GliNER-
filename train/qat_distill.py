"""
Hidden-state distillation + live diagnostics for E8 QAT.

Why this exists
---------------
Label-only QAT (the BCE loss through the GLiNER heads) gave the quantized encoder almost no useful
signal: the heads settle on "predict nothing" (validation loss 0.2785) while the damaged encoder output
(cosine to the FP encoder ~0.3-0.5, see diagnostics/qat_sensitivity_probe.py) is never repaired.

This module trains the QUANTIZED encoder to reproduce the FP encoder's last hidden state directly:
    kd = mean over real tokens of (1 - cos(student_h, teacher_h))   [+ optional relative MSE]
and prints, every N optimizer steps, numbers that tell us whether anything is learning:
    cos   mean cosine(student, teacher)   <- the progress metric; starts where the probe says, must RISE
    flip  share of quantized weights (sampled layers) whose codeword changed since the last report
          (0% for a long time = the quantized network is frozen = lr far too small / no gradient)
    gE8 / gOther   gradient norms of the E8 latent weights vs. everything else (0 or NaN = broken path)

How it hooks in (no change to the GLiNER model code): a forward hook on the HF encoder module captures the
exact (args, kwargs) GLiNER passes, replays them through a frozen FP teacher copy, and stores the loss;
finetune_gliner.py adds it to the task loss inside Trainer.compute_loss.

Selftest (no downloads):  python train\\qat_distill.py --selftest
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quantization"))
from e8_quant import E8QATLinear  # noqa: E402


def _last_hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _find_mask(args, kwargs, like):
    """Boolean (B, L) mask of real tokens; falls back to 'all tokens' if it cannot be found."""
    m = kwargs.get("attention_mask")
    if m is None:
        for a in args[1:]:
            if torch.is_tensor(a) and a.dim() == 2 and tuple(a.shape) == tuple(like.shape[:2]):
                m = a
                break
    if m is None:
        return torch.ones(like.shape[:2], dtype=torch.bool, device=like.device)
    return m.bool()


class EncoderDistiller:
    def __init__(self, student_enc: nn.Module, teacher_enc: nn.Module, mse_weight: float = 0.0,
                 teacher_dtype=None, disable_dropout: bool = True,
                 special_min_id=None, special_weight: float = 1.0):
        self.student = student_enc
        self.special_min_id = special_min_id      # token ids >= this are GLiNER's added prompt tokens (<<ENT>>, <<SEP>>)
        self.special_weight = float(special_weight)
        self._cos_sp = []
        self._nr, self._nr_sp = [], []          # ||student|| / ||teacher|| per token (cos ignores magnitude!)
        if disable_dropout:
            # The teacher runs without dropout; a student WITH dropout can never reach cos=1 and the diagnostic
            # baseline would read ~0.1 low.  (HF's Trainer calls model.train() every step, so set p=0, not .eval().)
            n_drop = 0
            for m in student_enc.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
                    n_drop += 1
                elif hasattr(m, "drop_prob"):          # DeBERTa's StableDropout
                    m.drop_prob = 0.0
                    n_drop += 1
            self.n_dropout_disabled = n_drop
        self.teacher = teacher_enc.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_dtype = teacher_dtype
        self._teacher_ready = False
        self.mse_weight = mse_weight
        self.kd = None
        self.nfwd = 0
        self._cos, self._kd = [], []
        self._snap = {}
        self.handle = student_enc.register_forward_hook(self._hook, with_kwargs=True)

    # -------------------------------------------------------------- hook
    def _hook(self, module, args, kwargs, output):
        if not (module.training and torch.is_grad_enabled()):
            return None
        s = _last_hidden(output)
        if not self._teacher_ready:
            kw = {"device": s.device}
            if self.teacher_dtype is not None:
                kw["dtype"] = self.teacher_dtype
            self.teacher.to(**kw)
            self._teacher_ready = True
        with torch.no_grad():
            t = _last_hidden(self.teacher(*args, **kwargs)).float()
        s = s.float()
        mask = _find_mask(args, kwargs, s)
        sm, tm = s[mask], t[mask]
        cos = F.cosine_similarity(sm, tm, dim=-1)
        sp_sel = None
        ids = args[0] if (args and torch.is_tensor(args[0])) else kwargs.get("input_ids")
        if self.special_min_id is not None and ids is not None and tuple(ids.shape) == tuple(mask.shape):
            sp_sel = ((ids >= self.special_min_id) & mask)[mask]
            if not bool(sp_sel.any()):
                sp_sel = None
        if sp_sel is not None and self.special_weight != 1.0:
            w = torch.ones_like(cos)
            w[sp_sel] = self.special_weight
            loss = ((1.0 - cos) * w).sum() / w.sum()
        else:
            loss = (1.0 - cos).mean()
        nr = (sm.detach().norm(dim=-1) / tm.norm(dim=-1).clamp_min(1e-12))
        self._nr.append(nr.mean())
        if sp_sel is not None:
            self._cos_sp.append(cos.detach()[sp_sel].mean())
            self._nr_sp.append(nr[sp_sel].mean())
        if self.mse_weight > 0:
            loss = loss + self.mse_weight * ((sm - tm).pow(2).sum() / tm.pow(2).sum().clamp_min(1e-12))
        self.kd = loss
        self.nfwd += 1
        self._cos.append(cos.detach().mean())
        self._kd.append(loss.detach())
        return None

    def take_loss(self):
        kd, self.kd = self.kd, None
        return kd

    # --------------------------------------------------------- diagnostics
    @torch.no_grad()
    def flip_share(self, every: int = 6):
        """Share of quantized values (every `every`-th E8 layer) that changed since the previous call."""
        flips = tot = 0
        i = -1
        for name, m in self.student.named_modules():
            if not isinstance(m, E8QATLinear):
                continue
            i += 1
            if i % every or m._q_cache is None:
                continue
            q = m._q_cache.detach().to(torch.bfloat16)
            prev = self._snap.get(name)
            if prev is not None and prev.shape == q.shape:
                flips += int((prev != q).sum().item())
                tot += q.numel()
            self._snap[name] = q.clone()
        return None if tot == 0 else flips / tot

    def should_report(self, every_fwd: int) -> bool:
        return every_fwd > 0 and self.nfwd > 0 and (self.nfwd == 1 or self.nfwd % every_fwd == 0)

    def report(self, grad_accum: int, log=print):
        if not self._cos:
            return
        cos = torch.stack(self._cos).mean().item()
        kd = torch.stack(self._kd).mean().item()
        n = len(self._cos)
        self._cos, self._kd = [], []
        flip = self.flip_share()
        fl = "n/a (first report)" if flip is None else f"{flip * 100:.3f}%"
        sp = f" | norm ratio {torch.stack(self._nr).mean().item():.3f}" if self._nr else ""
        self._nr = []
        if self._cos_sp:
            sp += (f" | PROMPT tokens: cos {torch.stack(self._cos_sp).mean().item():.4f}, "
                   f"norm ratio {torch.stack(self._nr_sp).mean().item():.3f}")
            self._cos_sp, self._nr_sp = [], []
        log(f"[qat-diag] step ~{self.nfwd // max(1, grad_accum):>5} | cos(student,FP) {cos:.4f}{sp} | kd {kd:.4f} "
            f"| weights flipped since last report {fl} | avg over {n} fwd", flush=True)


def grad_norms(model: nn.Module):
    """(E8 latent-weight grad norm, grad norm of every other parameter) -- NaN/0 here means a broken path."""
    e8 = {id(m.weight) for m in model.modules() if isinstance(m, E8QATLinear)}
    a = b = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        n = float(p.grad.detach().float().norm()) ** 2
        if id(p) in e8:
            a += n
        else:
            b += n
    return a ** 0.5, b ** 0.5


# ----------------------------------------------------------------------------- selftest
def selftest():
    from transformers import DebertaV2Config, DebertaV2Model
    torch.manual_seed(0)
    cfg = DebertaV2Config(vocab_size=200, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                          intermediate_size=128, relative_attention=True, position_buckets=16,
                          max_relative_positions=-1, pos_att_type=["p2c", "c2p"], max_position_embeddings=128,
                          position_biased_input=False, share_att_key=True, norm_rel_ebd="layer_norm")
    teacher = DebertaV2Model(cfg).eval()
    with torch.no_grad():
        for p in teacher.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.1)

    # 1) sanity: student == teacher  ->  cos == 1, loss == 0
    same = copy.deepcopy(teacher)
    same.train()
    d0 = EncoderDistiller(same, teacher)
    ids = torch.randint(5, 200, (4, 24)); mask = torch.ones_like(ids); mask[:, 20:] = 0
    same(ids, mask)                                  # positional call, like GLiNER's encoder
    assert d0.nfwd == 1 and d0.kd is not None
    assert d0.kd.item() < 1e-5, f"identical student must give ~0 loss, got {d0.kd.item()}"
    d0.handle.remove()

    # 2) eval mode / no_grad must NOT trigger the hook
    same.eval(); same(ids, mask)
    with torch.no_grad():
        same.train(); same(ids, mask)
    assert d0.nfwd == 1

    # 3) quantized student: loss must go down with training and codewords must flip
    student = copy.deepcopy(teacher)
    n = 0
    for layer in student.encoder.layer:
        for qual, mod in list(layer.named_modules()):
            if isinstance(mod, nn.Linear) and mod.in_features >= 64:
                parent_name, _, child = qual.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, child, E8QATLinear.from_linear(mod, scaling_mode="row_sigma_learnable", learnable=False))
                n += 1
    student.train()
    ids[:, :3] = 196                                  # pretend these are <<ENT>>/<<SEP>> prompt tokens
    d = EncoderDistiller(student, teacher, mse_weight=0.1, special_min_id=195, special_weight=3.0)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=3e-3)
    first = last = None
    flips_seen = False
    for step in range(60):
        student(ids, mask)
        c = float(d._cos[-1])
        kd = d.take_loss()
        opt.zero_grad(); kd.backward()
        g_e8, g_other = grad_norms(student)
        assert g_e8 > 0 and g_e8 == g_e8, "E8 latent weights must receive gradient"
        opt.step()
        if step % 10 == 0:
            if d.should_report(1):
                d.report(1)
            f = d.flip_share()
            flips_seen |= (f or 0) > 0
        if step == 0:
            first = c
        last = c
    assert last > first + 0.02, f"distillation did not raise cosine: {first:.3f} -> {last:.3f}"
    assert d.nfwd > 0
    print(f"layers quantized: {n} | cos {first:.3f} -> {last:.3f}")
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
