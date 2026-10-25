"""
e8_quant.py  (cached-quantization version)
────────────────────────────────────────────────────────────────
E8 lattice codebook (256-entry, one 8-bit index per 8-D sub-vector) + a
straight-through-estimator (STE) quantized Linear layer for QAT.

Scaling modes (unchanged from the original file):
  "block_max"           per-32-weight-block max scaling (matches export_e8_8bit())
  "row_sigma"           fixed per-row scale  beta = Y_max / (Cb * sigma_row)
  "row_sigma_learnable" EMA row statistics + learnable per-row theta_beta
                        (see E8QATLinear docstring)

WHAT CHANGED vs. the original (numerics of the quantizer itself are unchanged):
  1. The nearest-codeword search runs in fp32 with autocast disabled (under bf16
     autocast the old einsum ran in bf16 and could pick the wrong codeword) and in
     chunks (peak temp ~270 MB instead of ~600 MB+ per big layer).
  2. The quantized weight is CACHED. It is recomputed only after the weights can
     have changed, i.e. once per optimizer step instead of once per micro-batch.
     With --grad-accum 8 this removes 7 of every 8 full nearest-codeword searches
     over all 72 encoder Linears; in eval it is computed once in total.
     Invalidation = a global torch post-optimizer-step hook (works with fused
     AdamW, accelerate and HF Trainer, no callback needed) + the tensor version
     counter (catches load_state_dict / in-place edits).  If you edit weights
     through `.data` yourself, call invalidate_quant_cache(model).
     The autograd graph is the same STE graph as before (verified: identical
     outputs and gradients, see test_e8_equivalence.py).
  3. step_count is a plain Python int (no GPU sync), as in your local patch.

The DEFAULT codebook is NOT changed (codebook_bits=8), so the exported .bin / C++ LUT-GEMM stay in parity.

OPTIONAL 2-bit-per-weight codebook (codebook_bits=16, row_sigma_learnable only):
  65,536 entries = the 65,536 smallest E8 lattice points (shells of norm^2 0,2,4,6,8,10 in full plus a
  sign-symmetric part of shell 12), i.e. a 16-bit index per 8 weights = 2 bits/weight.  Measured on Gaussian
  weights its relative error is ~0.30 against ~0.56 for the 256-entry (1 bit/weight) codebook.
  Indices need 16 bits, so exports/C++ for this mode need uint16 indices.  Off by default.
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── cache invalidation ─────────────────────────────────────────────────
# Fused optimizers (e.g. AdamW(fused=True), the HF Trainer default on recent
# versions) update weights WITHOUT bumping PyTorch's tensor version counter, so the
# version counter alone is not safe. A global post-step hook bumps this counter after
# EVERY optimizer.step() of any optimizer (incl. accelerate / HF Trainer / fused),
# which invalidates all cached quantized weights.
_OPT_STEP = [0]


def _bump_opt_step(optimizer, args, kwargs):
    _OPT_STEP[0] += 1


_HOOK_HANDLE = None
try:
    from torch.optim.optimizer import register_optimizer_step_post_hook as _reg_hook
    _HOOK_HANDLE = _reg_hook(_bump_opt_step)
except Exception:   # very old torch: fall back to explicit invalidate_quant_cache()
    _HOOK_HANDLE = None

BLOCK_SIZE = 32     # weights per block (matches export_all_models.py)
SUBVEC_DIM = 8      # E8 lattice dimension
SUBVECS_PER_BLOCK = BLOCK_SIZE // SUBVEC_DIM   # = 4

# Global switch: set e8_quant.E8_CACHE_ENABLED = False to reproduce the old
# "re-quantize on every forward" behaviour (useful for A/B timing).
E8_CACHE_ENABLED = True
_SEARCH_CHUNK = 1 << 18   # sub-vectors per nearest-codeword chunk

# ── "row_sigma" (fixed, single-shot) scaling constants ─────────────────
ROW_SIGMA_CB = 5.0
ROW_SIGMA_CLIP = 2.0
ROW_SIGMA_DELTA0 = 1.5
ROW_SIGMA_Q = 4
ROW_SIGMA_M = 1


def set_row_sigma_params(delta_0: float = None, q: int = None, M: int = None,
                          Cb: float = None, clip_value: float = None):
    """Override the fixed row_sigma scaling parameters (e.g. from CLI args)."""
    global ROW_SIGMA_DELTA0, ROW_SIGMA_Q, ROW_SIGMA_M, ROW_SIGMA_CB, ROW_SIGMA_CLIP
    if delta_0 is not None:
        ROW_SIGMA_DELTA0 = delta_0
    if q is not None:
        ROW_SIGMA_Q = q
    if M is not None:
        ROW_SIGMA_M = M
    if Cb is not None:
        ROW_SIGMA_CB = Cb
    if clip_value is not None:
        ROW_SIGMA_CLIP = clip_value


def get_row_sigma_ymax() -> float:
    """Y_max = delta_0 * (q^M - 1) / 2 (paper's exact formula)."""
    return ROW_SIGMA_DELTA0 * (ROW_SIGMA_Q ** ROW_SIGMA_M - 1) / 2


# ── "row_sigma_learnable" constants (see original docstrings for sourcing) ──
LEARNABLE_DELTA0 = 1.5
LEARNABLE_Q = 4
LEARNABLE_M = 1
LEARNABLE_ETA = 0.1            # documentation only, not used in _bounds()
LEARNABLE_GAMMA_INV = 12.0     # documentation only, not used in _bounds()
LEARNABLE_EMA_MOMENTUM = 0.99
LEARNABLE_STAT_UPDATE_INTERVAL = 256
LEARNABLE_CLIP = 2.0
LEARNABLE_THETA_INIT = 0.0
LEARNABLE_RANGE_LOW = 0.5      # beta_min = 0.5 * anchor
LEARNABLE_RANGE_HIGH = 2.0     # beta_max = 2.0 * anchor
ANCHOR_STD_2BIT = 0.8          # 2-bit codebook: theta=0 gives beta = 1.25*anchor -> scaled-weight std ~1.0 (measured optimum)


# ─────────────────────────────────────────────────────────────
# Same E8 256-entry codebook as the C++ benchmark / export scripts.
# (Unchanged on purpose. NOTE: it is 1 zero vector + the 240 E8 roots + the first
#  15 norm-4 points in lexicographic order, so it is slightly asymmetric.)
# ─────────────────────────────────────────────────────────────
def build_e8_codebook_8bit() -> torch.Tensor:
    e8_points = []
    for p in itertools.product(range(-2, 3), repeat=8):
        if sum(p) % 2 == 0 and sum(x**2 for x in p) <= 4:
            e8_points.append(p)
    for p in itertools.product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        n = [x - 0.5 for x in p]
        if int(sum(n)) % 2 == 0 and sum(x**2 for x in p) <= 4:
            e8_points.append(p)

    e8_points = sorted(e8_points, key=lambda v: (sum(x**2 for x in v), v))
    codebook16 = torch.tensor(e8_points[:65536], dtype=torch.float32)
    if codebook16.shape[0] < 65536:
        pad = torch.zeros(65536 - codebook16.shape[0], 8)
        codebook16 = torch.cat([codebook16, pad], dim=0)

    codebook8 = codebook16[:256].clone()   # (256, 8)
    return codebook8


_CB16_CACHE = {}


def build_e8_codebook_16bit() -> torch.Tensor:
    """65,536-entry E8 codebook (2 bits/weight): all E8 points of norm^2 <= 10 (56,881 points) plus 8,655 of the
    60,480 norm^2=12 points, chosen as +-pairs from a seeded shuffle so the partial shell has no directional bias.
    (65,535 distinct entries; the last slot repeats the zero vector.)"""
    if "cb" in _CB16_CACHE:
        return _CB16_CACHE["cb"].clone()
    import numpy as np
    grids = {}
    ints = np.arange(-3, 4, dtype=np.int8)
    mesh = np.stack(np.meshgrid(*([ints] * 8), indexing="ij"), axis=-1).reshape(-1, 8)           # 7^8 x 8
    n_int = (mesh.astype(np.int16) ** 2).sum(1)
    keep = (mesh.sum(1) % 2 == 0) & (n_int <= 12)
    int_pts, int_n = mesh[keep].astype(np.float32), n_int[keep]
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    mesh = np.stack(np.meshgrid(*([halves] * 8), indexing="ij"), axis=-1).reshape(-1, 8)         # 6^8 x 8
    n_half = (mesh ** 2).sum(1)
    keep = (np.rint(mesh.sum(1)).astype(np.int64) % 2 == 0) & (n_half <= 12.001)
    half_pts, half_n = mesh[keep], np.rint(n_half[keep]).astype(np.int64)
    pts = np.concatenate([int_pts, half_pts])
    nrm = np.concatenate([int_n.astype(np.int64), half_n])
    full = pts[nrm <= 10]
    full = full[np.lexsort(np.rint(full * 2).astype(np.int64).T[::-1] .tolist() + [np.rint((full ** 2).sum(1)).astype(np.int64)])]
    shell = pts[nrm == 12]
    need = 65536 - len(full)
    # +-pairs: keep one representative per {p,-p} (lexicographically smaller), shuffle with a fixed seed
    key = np.rint(shell * 2).astype(np.int64)
    rep = np.array([tuple(k) <= tuple(-k) for k in key])
    reps = shell[rep]
    rng = np.random.RandomState(0)
    reps = reps[rng.permutation(len(reps))][: need // 2]
    chosen = np.concatenate([reps, -reps])
    cb = np.concatenate([full, chosen])
    if len(cb) < 65536:
        cb = np.concatenate([cb, np.zeros((65536 - len(cb), 8), dtype=np.float32)])
    cb = torch.tensor(cb[:65536], dtype=torch.float32)
    _CB16_CACHE["cb"] = cb
    return cb.clone()


def build_codebook(bits: int = 8) -> torch.Tensor:
    if bits == 8:
        return build_e8_codebook_8bit()
    if bits == 16:
        return build_e8_codebook_16bit()
    raise ValueError(f"codebook_bits must be 8 (1 bit/weight) or 16 (2 bits/weight), got {bits}")


# ─────────────────────────────────────────────────────────────
# Shared nearest-codeword search: fp32, autocast off, TF32 off, chunked.
# Returns uint8 indices (256 entries fit in one byte).
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def nearest_codeword_idx(subvecs: torch.Tensor, codebook: torch.Tensor,
                          norms: torch.Tensor) -> torch.Tensor:
    flat = subvecs.reshape(-1, SUBVEC_DIM)
    K = codebook.shape[0]
    idx_dtype = torch.uint8 if K <= 256 else torch.int32
    chunk = min(_SEARCH_CHUNK, max(256, (1 << 27) // K))      # keeps the (chunk x K) distance block <= 512 MB
    out = torch.empty(flat.shape[0], dtype=idx_dtype, device=flat.device)
    prev_tf32 = None
    if flat.is_cuda:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=flat.device.type, enabled=False):
            cb_t = codebook.float().t().contiguous()
            nr = norms.float()
            for i in range(0, flat.shape[0], chunk):
                d = flat[i:i + chunk].float() @ cb_t               # (n, K)
                d.mul_(-2.0).add_(nr)                              # |c|^2 - 2<x,c>
                out[i:i + chunk] = d.argmin(dim=1).to(idx_dtype)
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return out.reshape(subvecs.shape[:-1])


class E8NearestNeighborSTE(torch.autograd.Function):
    """Kept for API compatibility. subvecs: (..., 8) already-scaled blocks."""
    @staticmethod
    def forward(ctx, subvecs: torch.Tensor, codebook: torch.Tensor, norms: torch.Tensor):
        idx = nearest_codeword_idx(subvecs, codebook, norms)
        return codebook[idx.long()]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def e8_nearest_neighbor_ste(subvecs, codebook, norms):
    return E8NearestNeighborSTE.apply(subvecs, codebook, norms)


class E8QuantizeSTE(torch.autograd.Function):
    """STE quantization for "block_max" and "row_sigma" (same math as the original)."""
    @staticmethod
    def forward(ctx, W: torch.Tensor, codebook: torch.Tensor, norms: torch.Tensor,
                scaling_mode: str = "block_max"):
        rows, cols = W.shape
        pad = (-cols) % BLOCK_SIZE
        W_padded = F.pad(W, (0, pad)) if pad else W
        padded_cols = W_padded.shape[1]
        bpr = padded_cols // BLOCK_SIZE
        blocks = W_padded.reshape(rows * bpr, BLOCK_SIZE)

        if scaling_mode == "block_max":
            max_val = blocks.abs().amax(dim=1, keepdim=True)
            scale = torch.where(max_val > 1e-8, max_val / 2.0,
                                 torch.full_like(max_val, 1e-8))
        elif scaling_mode == "row_sigma":
            row_sigma = W_padded.std(dim=1, unbiased=False, keepdim=True)
            row_sigma = torch.clamp(row_sigma, min=1e-8)
            row_scale = (ROW_SIGMA_CB * row_sigma) / get_row_sigma_ymax()
            scale = row_scale.repeat_interleave(bpr, dim=0)
        else:
            raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")

        subvecs = (blocks / scale).reshape(-1, SUBVECS_PER_BLOCK, SUBVEC_DIM)
        idx = nearest_codeword_idx(subvecs, codebook, norms)
        recon_blocks = codebook[idx.long()].reshape(-1, BLOCK_SIZE) * scale

        W_quant_padded = recon_blocks.reshape(rows, padded_cols)
        W_quant = W_quant_padded[:, :cols] if pad else W_quant_padded
        if scaling_mode == "row_sigma":
            W_quant = torch.clamp(W_quant, -ROW_SIGMA_CLIP, ROW_SIGMA_CLIP)
        return W_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def e8_quantize_ste(W: torch.Tensor, codebook: torch.Tensor, norms: torch.Tensor,
                     scaling_mode: str = "block_max") -> torch.Tensor:
    return E8QuantizeSTE.apply(W, codebook, norms, scaling_mode)


class _UseCachedSTE(torch.autograd.Function):
    """Forward: return an already-computed quantized tensor `wq` (no copy, no search).
    Backward: identity gradient to `ref` (the differentiable thing it stands in for)."""
    @staticmethod
    def forward(ctx, ref: torch.Tensor, wq: torch.Tensor):
        return wq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ─────────────────────────────────────────────────────────────
# Drop-in replacement for nn.Linear with E8 STE weight quantization.
# ─────────────────────────────────────────────────────────────
class E8QATLinear(nn.Module):
    """
    scaling_mode="row_sigma_learnable" (mechanism from coset HNLQLinearQAT, with the
    bounds fix documented in the original file):
        EMA row stats every LEARNABLE_STAT_UPDATE_INTERVAL forwards (and on the first),
        anchor = Y_max / (Cb * sigma_ema); beta in [0.5*anchor, 2*anchor] picked by
        beta = beta_min + sigmoid(theta_beta) * (beta_max - beta_min)
        W_q = clip( nearest_E8(W * beta) / beta , -tau, tau )     (STE)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 scaling_mode: str = "block_max", learnable: bool = True, codebook_bits: int = 8):
        super().__init__()
        if codebook_bits != 8 and scaling_mode != "row_sigma_learnable":
            raise ValueError("codebook_bits=16 (2 bits/weight) is only supported with scaling_mode='row_sigma_learnable'")
        self.codebook_bits = codebook_bits
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_mode = scaling_mode
        self.learnable = learnable
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        codebook = build_codebook(codebook_bits)
        norms = (codebook ** 2).sum(dim=1)
        self.register_buffer("codebook", codebook, persistent=False)
        self.register_buffer("norms", norms, persistent=False)

        if scaling_mode == "row_sigma_learnable":
            if learnable:
                self.theta_beta = nn.Parameter(
                    torch.full((out_features,), LEARNABLE_THETA_INIT))
            else:
                self.register_buffer(
                    "theta_beta", torch.full((out_features,), LEARNABLE_THETA_INIT))
            self.register_buffer("sigma_ema", torch.ones(out_features))
            self.register_buffer("xmax_ema", torch.ones(out_features))
            self.step_count_int = 0       # python int: no GPU sync

        # quantization cache (plain attributes: not saved, not moved by .to())
        self._q_cache = None
        self._q_cache_key = None
        self.quant_calls = 0              # how many real nearest-codeword searches ran

    @classmethod
    def from_linear(cls, linear: nn.Linear, scaling_mode: str = "block_max",
                     learnable: bool = True, codebook_bits: int = 8) -> "E8QATLinear":
        """Convert an existing nn.Linear into an E8QATLinear, copying weights."""
        new_layer = cls(linear.in_features, linear.out_features,
                         bias=linear.bias is not None, scaling_mode=scaling_mode,
                         learnable=learnable, codebook_bits=codebook_bits)
        with torch.no_grad():
            new_layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_layer.bias.copy_(linear.bias)
            if scaling_mode == "row_sigma_learnable":
                # warm-start EMA stats from the real weight (see original note)
                new_layer.sigma_ema.copy_(new_layer.weight.std(dim=1) + 1e-8)
                new_layer.xmax_ema.copy_(new_layer.weight.abs().amax(dim=1) + 1e-8)
        return new_layer

    # ------------------------------------------------------------------ cache
    def invalidate_quant_cache(self):
        self._q_cache = None
        self._q_cache_key = None

    def _version_key(self):
        k = (_OPT_STEP[0], self.weight._version, self.weight.device, self.weight.dtype,
             self.weight.shape)
        if self.scaling_mode == "row_sigma_learnable":
            k += (getattr(self.theta_beta, "_version", 0),
                  getattr(self.sigma_ema, "_version", 0))
        return k

    # ------------------------------------------------------ learnable-mode helpers
    def _cache_stale(self, key):
        if (not E8_CACHE_ENABLED) or self._q_cache is None or self._q_cache_key != key:
            return True
        # A cache built under torch.inference_mode() is an inference tensor and
        # cannot be saved for backward. Rebuild it once when we leave inference mode.
        return self._q_cache.is_inference() and not torch.is_inference_mode_enabled()

    def _gather_stats(self):
        with torch.no_grad():
            m = LEARNABLE_EMA_MOMENTUM
            sigma = self.weight.std(dim=1) + 1e-8
            xmax = self.weight.abs().amax(dim=1) + 1e-8
            self.sigma_ema.mul_(m).add_((1 - m) * sigma)
            self.xmax_ema.mul_(m).add_((1 - m) * xmax)

    def _bounds(self):
        if self.codebook_bits == 8:
            anchor = get_row_sigma_ymax() / (ROW_SIGMA_CB * self.sigma_ema)       # unchanged 1-bit behaviour
        else:
            anchor = ANCHOR_STD_2BIT / self.sigma_ema
        return anchor * LEARNABLE_RANGE_LOW, anchor * LEARNABLE_RANGE_HIGH

    def _learnable_quantize_scaled(self, W_scaled: torch.Tensor) -> torch.Tensor:
        """nearest_E8 of already-scaled weights (no grad), shape (out, in)."""
        with torch.no_grad():
            pad = (-self.in_features) % SUBVEC_DIM
            Wp = F.pad(W_scaled, (0, pad)) if pad else W_scaled
            subvecs = Wp.reshape(self.out_features, -1, SUBVEC_DIM)
            idx = nearest_codeword_idx(subvecs, self.codebook, self.norms)
            rec = self.codebook[idx.long()].reshape(self.out_features, -1)
            self.quant_calls += 1
            return rec[:, :self.in_features] if pad else rec

    def _forward_learnable(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight
        if self.training:
            self.step_count_int += 1
            if (self.step_count_int % LEARNABLE_STAT_UPDATE_INTERVAL) == 0 or self.step_count_int == 1:
                self._gather_stats()

        beta_min, beta_max = self._bounds()
        beta = (beta_min + torch.sigmoid(self.theta_beta) * (beta_max - beta_min)).view(-1, 1)
        W_scaled = W * beta

        key = self._version_key()
        if self._cache_stale(key):
            self._q_cache = self._learnable_quantize_scaled(W_scaled.detach())
            self._q_cache_key = key

        recon = _UseCachedSTE.apply(W_scaled, self._q_cache)    # STE: identity grad to W_scaled
        W_quant = torch.clamp(recon / beta, -LEARNABLE_CLIP, LEARNABLE_CLIP)
        return F.linear(x, W_quant, self.bias)

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaling_mode == "row_sigma_learnable":
            return self._forward_learnable(x)

        key = self._version_key()
        if self._cache_stale(key):
            with torch.no_grad():
                self._q_cache = e8_quantize_ste(self.weight.detach(), self.codebook,
                                                  self.norms, self.scaling_mode)
            self._q_cache_key = key
            self.quant_calls += 1
        w_q = self._q_cache
        if torch.is_grad_enabled() and self.weight.requires_grad:
            w_q = _UseCachedSTE.apply(self.weight, w_q)         # STE: identity grad to latent weight
        return F.linear(x, w_q, self.bias)

    def extra_repr(self):
        s = (f"in_features={self.in_features}, out_features={self.out_features}, "
             f"e8_qat=True, codebook_bits={self.codebook_bits}, scaling_mode={self.scaling_mode!r}")
        if self.scaling_mode == "row_sigma_learnable":
            s += f", learnable={self.learnable}"
        return s


def invalidate_quant_cache(model: nn.Module):
    """Drop every cached quantized weight (e.g. before saving or after loading a state_dict)."""
    for m in model.modules():
        if isinstance(m, E8QATLinear):
            m.invalidate_quant_cache()


def count_quant_calls(model: nn.Module) -> int:
    return sum(m.quant_calls for m in model.modules() if isinstance(m, E8QATLinear))
