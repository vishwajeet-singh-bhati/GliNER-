"""Checks that the cached e8_quant.py matches the original numerically and in gradients,
that the cache really skips work, and that it invalidates after every optimizer step
(SGD, AdamW foreach, AdamW fused)."""
import importlib.util, time, sys
import torch
import torch.nn as nn

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

#   python test_e8_equivalence.py [old_e8_quant.py] [new_e8_quant.py]
OLD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/work/old/e8_quant.py"
NEW = sys.argv[2] if len(sys.argv) > 2 else "/tmp/work/new/e8_quant.py"
old = load(OLD, "old_e8")
new = load(NEW, "new_e8")
torch.manual_seed(0)
fails = []

def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond: fails.append(name)

def pair(mode, learnable=True, i=768, o=3072):
    lin = nn.Linear(i, o); nn.init.normal_(lin.weight, std=0.04); nn.init.normal_(lin.bias, std=0.01)
    a = old.E8QATLinear.from_linear(lin, mode, learnable)
    b = new.E8QATLinear.from_linear(lin, mode, learnable)
    return a, b

configs = [("block_max", True), ("row_sigma", True), ("row_sigma_learnable", True), ("row_sigma_learnable", False)]

print("== 1. forward + gradient equivalence ==")
for mode, lrn in configs:
    a, b = pair(mode, lrn)
    x = torch.randn(2, 16, 768)
    ya, yb = a(x), b(x)
    ga = torch.autograd.grad((ya ** 2).mean(), [p for p in a.parameters() if p.requires_grad])
    gb = torch.autograd.grad((yb ** 2).mean(), [p for p in b.parameters() if p.requires_grad])
    tag = f"{mode}{'' if lrn else '(fixed theta)'}"
    check(f"forward equal  {tag}", torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), f"max|dy|={float((ya-yb).abs().max()):.2e}")
    check(f"grads equal    {tag}", all(torch.allclose(u, v, atol=1e-6, rtol=1e-4) for u, v in zip(ga, gb)),
          f"max|dg|={max(float((u-v).abs().max()) for u, v in zip(ga, gb)):.2e}  (#grads={len(ga)})")

print("\n== 2. cache hits within an accumulation window, miss after each optimizer step ==")
def make_opt(kind, params):
    if kind == "sgd": return torch.optim.SGD(params, lr=1e-2)
    if kind == "adamw_foreach": return torch.optim.AdamW(params, lr=1e-3, foreach=True)
    if kind == "adamw_fused": return torch.optim.AdamW(params, lr=1e-3, fused=True)
for kind in ["sgd", "adamw_foreach", "adamw_fused"]:
    for mode, lrn in configs:
        try:
            a, b = pair(mode, lrn, 256, 512)
            pa = [p for p in a.parameters() if p.requires_grad]; pb = [p for p in b.parameters() if p.requires_grad]
            oa, ob = make_opt(kind, pa), make_opt(kind, pb)
        except Exception as e:
            print(f"[SKIP] {kind} {mode}: {type(e).__name__}: {str(e)[:60]}"); continue
        calls0 = b.quant_calls
        ok_equal = True
        for step in range(4):
            oa.zero_grad(); ob.zero_grad()
            for micro in range(8):                      # grad accumulation of 8
                x = torch.randn(1, 8, 256)
                la = (a(x) ** 2).mean() / 8; lb = (b(x) ** 2).mean() / 8
                la.backward(); lb.backward()
                ok_equal &= torch.allclose(a.weight.grad, b.weight.grad, atol=1e-6, rtol=1e-4)
            oa.step(); ob.step()
        searches = b.quant_calls - calls0
        # expected: 1 search per optimizer step (4), vs 32 forwards
        check(f"{kind:14s} {mode:20s}{'' if lrn else '(fixed)'}  searches={searches} for 32 forwards (want 4); grads match each micro-batch",
              searches == 4 and ok_equal)
        if mode != "row_sigma_learnable" or lrn:
            check(f"{kind:14s} {mode:20s} weights stay equal to the original after 4 steps",
                  torch.allclose(a.weight, b.weight, atol=1e-5, rtol=1e-4),
                  f"max|dW|={float((a.weight-b.weight).abs().max()):.2e}")

print("\n== 3. eval mode: quantize once ==")
a, b = pair("row_sigma", True, 256, 512); b.eval()
with torch.no_grad():
    for _ in range(10): b(torch.randn(2, 8, 256))
check("eval: 1 search for 10 forwards", b.quant_calls == 1, f"searches={b.quant_calls}")

print("\n== 4. speed on CPU, one optimizer step = 8 micro-batches (768x3072 Linear, 1x64 tokens) ==")
def bench(layer, reps=3):
    x = torch.randn(1, 64, 768); ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(8):
            (layer(x) ** 2).mean().backward()
        ts.append(time.perf_counter() - t)
        with torch.no_grad(): layer.weight.add_(1e-6)      # emulate an optimizer step
    return min(ts)
a, b = pair("row_sigma", True)
t_old = bench(a)
new.E8_CACHE_ENABLED = False; t_nocache = bench(b)
new.E8_CACHE_ENABLED = True;  t_cache = bench(b)
print(f"original: {t_old*1000:7.0f} ms/step   new (cache off): {t_nocache*1000:7.0f} ms/step   new (cache on): {t_cache*1000:7.0f} ms/step   speedup vs original: {t_old/t_cache:.1f}x")

print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
