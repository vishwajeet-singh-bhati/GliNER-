"""Verifies a fusion checkpoint really uses the structural stream (not silently skipped).
  python fusion\\check_fusion_active.py <checkpoint_dir> [--base-model urchade/gliner_multi_pii-v1]
Checks: (1) fusion keys in checkpoint, (2) gate bias moved from its init (training touched it),
(3) fusion module is called during inference, (4) mean gate value, (5) output changes if char ids are removed."""
import argparse, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "eval"))
from char_fusion import FusedSpanRep
import evaluate_gliner_audit as ev

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
ap.add_argument("--init-gate-bias", type=float, default=4.0)
a = ap.parse_args()

w = ev._find_weights_file(Path(a.ckpt)); sd = ev._load_state_dict(w)
fk = [k for k in sd if "span_rep_layer.fusion." in k]
print(f"[1] fusion tensors in checkpoint: {len(fk)}"); assert fk, "NO fusion keys: checkpoint is not a fusion model"
gb = next(v for k, v in sd.items() if k.endswith("fusion.gate.bias")).float()
print(f"[2] gate bias mean {gb.mean():.4f} (init {a.init_gate_bias}); max |change| {(gb - a.init_gate_bias).abs().max():.5f}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = ev.load_ordinary_checkpoint(a.ckpt, a.base_model, dev)
core = m.model; layer = core.span_rep_layer
assert isinstance(layer, FusedSpanRep), "loaded model has no FusedSpanRep"
calls = []
layer.fusion.register_forward_hook(lambda mod, i, o: calls.append(float(mod.last_gate_mean)))
text = "Call John Smith on +91 98765 43210 or mail john.smith@example.com, card 4111 1111 1111 1111, id 123e4567-e89b-12d3-a456-426614174000"
labels = ["person name", "phone number", "email", "credit card number", "uuid"]
with torch.no_grad():
    p1 = m.predict_entities(text, labels, threshold=0.3)
print(f"[3] fusion calls during inference: {len(calls)}"); assert calls, "FUSION NOT CALLED (char_ids not reaching the model)"
print(f"[4] mean gate (sem weight) at inference: {sum(calls)/len(calls):.4f}")
orig = layer.fusion.forward                      # compare against semantic-only (fusion bypassed)
layer.fusion.forward = lambda sem, ch: sem
with torch.no_grad():
    p2 = m.predict_entities(text, labels, threshold=0.3)
layer.fusion.forward = orig
print("[5] predictions with fusion   :", [(e['text'], e['label'], round(e['score'], 3)) for e in p1])
print("    predictions semantic-only :", [(e['text'], e['label'], round(e['score'], 3)) for e in p2])
print("OK: fusion is present and active" if calls else "FAIL")
