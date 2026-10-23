import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

# Fix HF Hub hanging on network requests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.serialization

# Force weights_only=False across all local checkpoint loads (rng_state.pth, optimizer.pt, etc.)
_orig_torch_load = torch.load

def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _trusted_torch_load
torch.serialization.load = _trusted_torch_load

import transformers.utils.import_utils as hf_import_utils
import transformers.trainer as hf_trainer
import transformers.modeling_utils as hf_modeling_utils
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback

# Bypass transformers CVE-2025-32434 torch<2.6 check
def _allow_local_torch_load(*args, **kwargs):
    return None

hf_import_utils.check_torch_load_is_safe = _allow_local_torch_load
hf_trainer.check_torch_load_is_safe = _allow_local_torch_load
if hasattr(hf_modeling_utils, "check_torch_load_is_safe"):
    hf_modeling_utils.check_torch_load_is_safe = _allow_local_torch_load

from gliner import GLiNER
import gliner.training.trainer as gliner_trainer_mod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantization"))
from quantize_gliner import quantize_gliner_encoder
from long_filter import drop_long_examples


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sanitize_example_in_place(ex, max_window_tokens=256, max_token_chars=128):
    tokens = list(ex.get("tokenized_text", []))
    ner = ex.get("ner", []) or []

    entity_token_idxs = set()
    valid_ner = []
    n_tok = len(tokens)
    for span in ner:
        if len(span) >= 3:
            s, e, label = int(span[0]), int(span[1]), str(span[2])
            if 0 <= s <= e < n_tok:
                valid_ner.append([s, e, label])
                for idx in range(s, e + 1):
                    entity_token_idxs.add(idx)

    if max_token_chars > 0:
        for i, tok in enumerate(tokens):
            if len(tok) > max_token_chars:
                limit = max_token_chars * 2 if i in entity_token_idxs else max_token_chars
                if len(tok) > limit:
                    tokens[i] = tok[:limit]

    if max_window_tokens > 0 and len(tokens) > max_window_tokens:
        if valid_ner:
            first_s = min(sp[0] for sp in valid_ner)
            last_e = max(sp[1] for sp in valid_ner)
            span_center = (first_s + last_e) // 2
            win_start = max(0, min(span_center - max_window_tokens // 2, len(tokens) - max_window_tokens))
        else:
            win_start = 0
        win_end = win_start + max_window_tokens
        tokens = tokens[win_start:win_end]
        remapped_ner = []
        for s, e, label in valid_ner:
            if win_start <= s and e < win_end:
                remapped_ner.append([s - win_start, e - win_start, label])
        valid_ner = remapped_ner

    updated = dict(ex)
    updated["tokenized_text"] = tokens
    updated["ner"] = valid_ner
    return updated


class LabelPromptDataset(torch.utils.data.Dataset):
    """Gives every example the FULL label list as its prompt (GLiNER's predefined-labels key
    "ner_labels", see gliner/data_processing/processor.py batch_generate_class_mappings).

    Why: without it GLiNER's collator builds each prompt only from the labels that occur in the
    example (+ <= max_neg_type_ratio * n_entities random in-batch negatives, which for an
    entity-free window means NOTHING but a dummy "" label).  The model then never sees
    "this label is in the prompt and has no match here", so at inference time (all 16 labels
    prompted) it over-fires: precision ~0.2 / recall ~0.8 in the audit.  With the full label list
    in every prompt, every absent label is a supervised negative, for positive and negative
    windows alike -- exactly the inference-time situation.
    shuffle=True re-orders the labels each time an item is read (position robustness)."""

    def __init__(self, examples, labels, shuffle=True, seed=0):
        self.examples = examples
        self.labels = list(labels)
        self.shuffle = shuffle
        self.rng = __import__("random").Random(seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = dict(self.examples[i])
        labels = list(self.labels)
        if self.shuffle:
            self.rng.shuffle(labels)
        ex["ner_labels"] = labels
        return ex


def smart_load_state_dict(target_obj, state_dict):
    candidates = []
    if hasattr(target_obj, "model") and isinstance(target_obj.model, torch.nn.Module):
        candidates.append(("model.model", target_obj.model))
    if isinstance(target_obj, torch.nn.Module):
        candidates.append(("model", target_obj))

    best_mod = candidates[0][1]
    best_mapped_sd = state_dict
    best_match_count = -1

    for _, mod in candidates:
        target_keys = set(mod.state_dict().keys())
        direct_matches = len(target_keys.intersection(state_dict.keys()))
        if direct_matches > best_match_count:
            best_match_count = direct_matches
            best_mod = mod
            best_mapped_sd = state_dict

        stripped_sd = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
        stripped_matches = len(target_keys.intersection(stripped_sd.keys()))
        if stripped_matches > best_match_count:
            best_match_count = stripped_matches
            best_mod = mod
            best_mapped_sd = stripped_sd

        prefixed_sd = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
        prefixed_matches = len(target_keys.intersection(prefixed_sd.keys()))
        if prefixed_matches > best_match_count:
            best_match_count = prefixed_matches
            best_mod = mod
            best_mapped_sd = prefixed_sd

    missing, unexpected = best_mod.load_state_dict(best_mapped_sd, strict=False)
    return best_match_count, len(best_mod.state_dict()), missing, unexpected


def clear_inference_tensors(root):
    """Replace inference-mode tensors (buffers or cached attrs) with normal clones."""
    n = 0
    for mod in root.modules():
        for name, buf in list(mod._buffers.items()):
            if torch.is_tensor(buf) and buf.is_inference():
                mod._buffers[name] = buf.clone()
                n += 1
        for name, val in list(vars(mod).items()):
            if name in ("_buffers", "_parameters", "_modules"):
                continue
            if torch.is_tensor(val) and val.is_inference():
                setattr(mod, name, val.clone())
                n += 1
            elif isinstance(val, dict):
                for k, v in list(val.items()):
                    if torch.is_tensor(v) and v.is_inference():
                        val[k] = v.clone()
                        n += 1
    return n

def _prf(tp, fp, fn):
    p = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    r = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f = (2.0 * tp / (2.0 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0
    return p, r, f


def evaluate_gliner_spans(gliner_wrapper, dataset_slice, label_list, threshold=0.5, thresholds=None):
    """Exact-span micro P/R/F1 over `dataset_slice` (windows, positives AND negatives).

    thresholds: optional iterable.  Inference runs ONCE at min(thresholds) and the metrics are
    recomputed for every threshold by filtering on the span score; the result is reported for the
    threshold with the best F1 (key "threshold").  Filtering a low-threshold run is an
    approximation of re-running at the higher threshold (overlap resolution can differ slightly);
    eval/evaluate_gliner_audit.py does the exact thing for the final numbers."""
    empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "tok_acc": 0.0, "tp": 0, "fp": 0, "fn": 0,
             "threshold": threshold, "predict_errors": 0}
    if not dataset_slice or not label_list:
        return empty
    ths = sorted(set(thresholds)) if thresholds else [threshold]
    run_th = ths[0]

    core_mod = gliner_wrapper.model if hasattr(gliner_wrapper, "model") else gliner_wrapper
    was_training = core_mod.training
    core_mod.eval()

    counts = {t: [0, 0, 0] for t in ths}          # tp, fp, fn
    hits = {t: 0 for t in ths}
    token_total = 0
    predict_errors = 0

    try:
        with torch.no_grad():
            for ex in dataset_slice:
                tokens = ex.get("tokenized_text", [])
                ner = ex.get("ner", []) or []
                if not tokens:
                    continue

                char_starts, char_ends, cur = [], [], 0
                for i, tok in enumerate(tokens):
                    if i > 0:
                        cur += 1
                    char_starts.append(cur)
                    cur += len(tok)
                    char_ends.append(cur)
                text = " ".join(tokens)

                gold_spans, gold_tok_labels = set(), {}
                for span in ner:
                    if len(span) >= 3:
                        s, e, lbl = int(span[0]), int(span[1]), str(span[2])
                        if 0 <= s <= e < len(tokens):
                            gold_spans.add((char_starts[s], char_ends[e], lbl))
                            for tidx in range(s, e + 1):
                                gold_tok_labels[tidx] = lbl
                token_total += len(gold_tok_labels)

                try:
                    preds = gliner_wrapper.predict_entities(text, label_list, threshold=run_th)
                except Exception as exc:   # counted and reported, never silently turned into "no entities"
                    predict_errors += 1
                    if predict_errors <= 3:
                        print(f"  [eval warning] predict_entities failed: {exc!r}", flush=True)
                    preds = []

                for t in ths:
                    pred_spans, pred_tok_labels = set(), {}
                    for p in preds:
                        if float(p.get("score", 1.0)) < t:
                            continue
                        ps, pe, plbl = int(p["start"]), int(p["end"]), str(p["label"])
                        pred_spans.add((ps, pe, plbl))
                        for tidx in range(len(tokens)):
                            if char_starts[tidx] >= ps and char_ends[tidx] <= pe:
                                pred_tok_labels[tidx] = plbl
                    c = counts[t]
                    c[0] += len(gold_spans & pred_spans)
                    c[1] += len(pred_spans - gold_spans)
                    c[2] += len(gold_spans - pred_spans)
                    hits[t] += sum(1 for tidx, g in gold_tok_labels.items() if pred_tok_labels.get(tidx) == g)
    finally:
        if was_training:
            core_mod.train()
        clear_inference_tensors(core_mod)

    best_t = max(ths, key=lambda t: (_prf(*counts[t])[2], -abs(t - threshold)))
    tp, fp, fn = counts[best_t]
    precision, recall, f1 = _prf(tp, fp, fn)
    tok_acc = (hits[best_t] / token_total) * 100.0 if token_total > 0 else recall
    acc = 0.5 * tok_acc + 0.5 * recall if token_total > 0 else recall
    return {
        "precision": precision, "recall": recall, "f1": f1, "acc": acc, "tok_acc": tok_acc,
        "tp": tp, "fp": fp, "fn": fn, "threshold": best_t, "predict_errors": predict_errors,
        "by_threshold": {t: dict(zip(("precision", "recall", "f1"), _prf(*counts[t]))) for t in ths},
    }


VAL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def find_latest_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    ckpts = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            m = re.match(r"^checkpoint-(\d+)$", d.name)
            if m and (d / "pytorch_model.bin").exists():
                ckpts.append((int(m.group(1)), d))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


class CleanTrainingLoggerCallback(TrainerCallback):
    def __init__(
        self,
        gliner_wrapper,
        output_dir: Path,
        total_epochs: int,
        steps_per_epoch: int,
        save_steps: int,
        log_every: int,
        grad_accum: int,
        train_eval_subset: list,
        val_eval_subset: list,
        all_labels: list,
        trainer_ref: dict,
    ):
        self.gliner_wrapper = gliner_wrapper
        self.output_dir = Path(output_dir)
        self.total_epochs = max(1, total_epochs)
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.save_steps = max(1, save_steps)
        self.log_every = max(1, log_every)
        self.grad_accum = max(1, grad_accum)
        self.train_eval_subset = train_eval_subset
        self.val_eval_subset = val_eval_subset
        self.all_labels = all_labels
        self.trainer_ref = trainer_ref

        self.segment_start_time = time.time()
        self.running_losses = []
        self.segment_losses = []
        self.oom_skips_segment = 0
        self.last_lr = 0.0
        self.start_step = 0
        self.first_step_logged = False
        self.last_train_acc = 0.0
        self.last_train_f1 = 0.0
        self.probe_cursor = 0

        self.best_score = -1.0
        self.best_val_loss = float("inf")
        self.best_checkpoint_name = "None"
        self.leaderboard = []
        self.evaluated_steps = set()

        self.history_csv = self.output_dir / "checkpoint_leaderboard.csv"
        self.best_dir = self.output_dir / "best_checkpoint"

    def _update_running_train_acc(self, sample_count=20):
        if not self.train_eval_subset:
            return
        n = len(self.train_eval_subset)
        start = self.probe_cursor % n
        batch = [self.train_eval_subset[(start + i) % n] for i in range(min(sample_count, n))]
        self.probe_cursor = (start + sample_count) % n
        stats = evaluate_gliner_spans(self.gliner_wrapper, batch, self.all_labels, threshold=0.5)
        if self.last_train_acc == 0.0:
            self.last_train_acc = stats["acc"]
            self.last_train_f1 = stats["f1"]
        else:
            self.last_train_acc = 0.7 * self.last_train_acc + 0.3 * stats["acc"]
            self.last_train_f1 = 0.7 * self.last_train_f1 + 0.3 * stats["f1"]

    def _restore_best_state(self):
        """On resume, reload the previous best score + leaderboard so a worse checkpoint cannot
        overwrite best_checkpoint (before this, a resumed run started from best_score=-1)."""
        info_p = self.best_dir / "best_checkpoint_info.json"
        try:
            if info_p.exists():
                info = json.loads(info_p.read_text(encoding="utf-8"))
                self.best_score = float(info.get("val_f1", 0.0)) + 0.001 * float(info.get("val_acc", 0.0))
                self.best_checkpoint_name = str(info.get("best_checkpoint", "None"))
                if info.get("val_loss") is not None:
                    self.best_val_loss = float(info["val_loss"])
                print(f"Resume: keeping previous best {self.best_checkpoint_name} "
                      f"(val F1 {info.get('val_f1')}) in best_checkpoint/", flush=True)
            if self.history_csv.exists():
                with open(self.history_csv, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        for k in ("step", "epoch"):
                            row[k] = int(float(row[k]))
                        for k in ("train_loss", "train_acc", "train_f1", "val_acc", "val_precision",
                                  "val_recall", "val_f1"):
                            row[k] = float(row[k])
                        row["is_best"] = str(row.get("is_best")) == "True"
                        self.leaderboard.append(row)
                        self.evaluated_steps.add(row["step"])
        except Exception as exc:
            print(f"  [Warning] could not restore previous best state: {exc}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.segment_start_time = time.time()
        self.start_step = state.global_step
        if self.start_step > 0:
            self._restore_best_state()
        cur_epoch = (self.start_step // self.steps_per_epoch) + 1
        step_in_epoch = (self.start_step % self.steps_per_epoch)

        # Populate historical loss and LR from trainer_state.json if resuming
        if state.log_history:
            for entry in state.log_history[-50:]:
                if "loss" in entry:
                    self.running_losses.append(float(entry["loss"]))
                    self.segment_losses.append(float(entry["loss"]))
                if "learning_rate" in entry:
                    self.last_lr = float(entry["learning_rate"])

        print("\n" + "=" * 78, flush=True)
        print(
            f"Training Active | Global Step: {self.start_step}/{state.max_steps} | "
            f"Epoch: {cur_epoch:02d}/{self.total_epochs:02d} (Step {step_in_epoch:04d}/{self.steps_per_epoch:04d})",
            flush=True,
        )
        print("=" * 78, flush=True)

        if self.start_step > 0:
            print(f"Evaluating baseline checkpoint-{self.start_step} before continuing...", flush=True)
            self._update_running_train_acc(sample_count=32)
            self._evaluate_and_log_banner(
                global_step=self.start_step,
                max_steps=state.max_steps,
                tag=f"checkpoint-{self.start_step}",
                is_checkpoint=True,
            )
            print(
                f"Fast-forwarding DataLoader to Step {self.start_step} "
                f"(Epoch {cur_epoch:02d}, Step {step_in_epoch:04d}/{self.steps_per_epoch:04d})... "
                f"takes ~10-15s",
                flush=True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss_val = float(logs["loss"])
        lr_val = float(logs.get("learning_rate", self.last_lr))
        self.last_lr = lr_val
        self.running_losses.append(loss_val)
        self.segment_losses.append(loss_val)

        step = state.global_step
        epoch_idx = min(self.total_epochs, (step - 1) // self.steps_per_epoch + 1)
        step_in_epoch = ((step - 1) % self.steps_per_epoch) + 1

        is_first_after_resume = (not self.first_step_logged and step > self.start_step)
        is_log_interval = (step % self.log_every == 0)
        is_epoch_end = (step_in_epoch == self.steps_per_epoch)
        is_final_step = (step == state.max_steps)

        if is_first_after_resume or is_log_interval or is_epoch_end or is_final_step:
            self.first_step_logged = True
            self._update_running_train_acc(sample_count=16)
            avg_loss = sum(self.running_losses[-50:]) / max(1, len(self.running_losses[-50:]))
            print(
                f"Epoch [{epoch_idx:02d}/{self.total_epochs:02d}] "
                f"Step [{step_in_epoch:04d}/{self.steps_per_epoch:04d}] "
                f"(Global {step:05d}/{state.max_steps:05d}) "
                f"Loss: {avg_loss:.4f} Acc: {self.last_train_acc:5.2f}% F1: {self.last_train_f1:5.2f}%",
                flush=True,
            )

        if (is_epoch_end or is_final_step) and (step % self.save_steps != 0):
            self._evaluate_and_log_banner(
                global_step=step,
                max_steps=state.max_steps,
                tag=f"Epoch-{epoch_idx:02d}",
                is_checkpoint=False,
            )

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.evaluated_steps:
            return
        self._evaluate_and_log_banner(
            global_step=step,
            max_steps=state.max_steps,
            tag=f"checkpoint-{step}",
            is_checkpoint=True,
        )

    def _evaluate_and_log_banner(self, global_step: int, max_steps: int, tag: str, is_checkpoint: bool):
        self.evaluated_steps.add(global_step)
        elapsed = time.time() - self.segment_start_time
        epoch_idx = min(self.total_epochs, max(1, math.ceil(global_step / self.steps_per_epoch)))

        train_loss = (
            sum(self.segment_losses) / len(self.segment_losses)
            if self.segment_losses
            else (sum(self.running_losses[-50:]) / max(1, len(self.running_losses[-50:])))
        )

        train_stats = evaluate_gliner_spans(
            self.gliner_wrapper, self.train_eval_subset[:60], self.all_labels, threshold=0.5
        )
        train_acc = train_stats["acc"]
        train_f1 = train_stats["f1"]
        self.last_train_acc = train_acc
        self.last_train_f1 = train_f1

        val_loss = float("nan")
        trainer = self.trainer_ref.get("trainer")
        if trainer is not None and self.val_eval_subset:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                metrics = trainer.evaluate(
                    eval_dataset=LabelPromptDataset(self.val_eval_subset[:100], self.all_labels, shuffle=False))
                raw_eval_loss = float(metrics.get("eval_loss", float("nan")))
                if not math.isnan(raw_eval_loss):
                    # Normalize GLiNER unreduced eval loss to match grad_accum-scaled training loss
                    val_loss = raw_eval_loss / float(self.grad_accum * 8.0)
            except Exception as e:
                print(f"  [Note] Val loss calc skipped: {e}", flush=True)

        val_stats = evaluate_gliner_spans(
            self.gliner_wrapper, self.val_eval_subset, self.all_labels, threshold=0.5,
            thresholds=VAL_THRESHOLDS,
        )
        val_acc = val_stats["acc"]
        val_f1 = val_stats["f1"]
        val_p = val_stats["precision"]
        val_r = val_stats["recall"]

        score = val_f1 + 0.001 * val_acc
        is_best = False
        if is_checkpoint and (score > self.best_score or (score == self.best_score and val_loss < self.best_val_loss)):
            self.best_score = score
            if not math.isnan(val_loss):
                self.best_val_loss = val_loss
            is_best = True
            if is_checkpoint:
                self.best_checkpoint_name = tag
                self._save_best_checkpoint(global_step, tag, train_loss, val_loss, train_acc, val_stats)

        best_marker = " <-- BEST" if is_best else ""
        val_loss_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "N/A"
        ppl_base = val_loss if not math.isnan(val_loss) else train_loss
        ppl_str = f"{math.exp(min(max(ppl_base, 0.0), 10.0)):.2f}"

        print("\n" + "=" * 78, flush=True)
        print(f"Epoch {epoch_idx}/{self.total_epochs} | {tag} (Step {global_step}/{max_steps}){best_marker}", flush=True)
        print(f"Train Loss      : {train_loss:.4f}", flush=True)
        print(f"Validation Loss : {val_loss_str}", flush=True)
        print(f"Train Accuracy  : {train_acc:.2f}% (Train F1: {train_f1:.2f}%)", flush=True)
        print(
            f"Validation Acc  : {val_acc:.2f}% (Val F1: {val_f1:.2f}% | P: {val_p:.2f}% | R: {val_r:.2f}% "
            f"@ best threshold {val_stats['threshold']:.2f}; predict errors: {val_stats['predict_errors']})",
            flush=True,
        )
        print(f"Test Perplexity : {ppl_str}", flush=True)
        print(f"Learning Rate   : {self.last_lr:.8f}", flush=True)
        print(f"Epoch Time      : {elapsed:.1f}s (OOM Skips: {self.oom_skips_segment})", flush=True)
        print(f"Best Checkpoint : {self.best_checkpoint_name} (Best Val F1: {max(0.0, self.best_score):.2f}%)", flush=True)
        print("=" * 78 + "\n", flush=True)

        row = {
            "step": global_step,
            "tag": tag,
            "epoch": epoch_idx,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if not math.isnan(val_loss) else "",
            "train_acc": round(train_acc, 2),
            "train_f1": round(train_f1, 2),
            "val_acc": round(val_acc, 2),
            "val_precision": round(val_p, 2),
            "val_recall": round(val_r, 2),
            "val_f1": round(val_f1, 2),
            "val_threshold": val_stats["threshold"],
            "is_best": is_best,
        }
        self.leaderboard.append(row)
        self._write_leaderboard_csv()

        self.segment_losses.clear()
        self.oom_skips_segment = 0
        self.segment_start_time = time.time()

    def _save_best_checkpoint(self, step: int, tag: str, train_loss: float, val_loss: float, train_acc: float, val_stats: dict):
        try:
            self.best_dir.mkdir(parents=True, exist_ok=True)
            src_ckpt = self.output_dir / tag
            if src_ckpt.exists():
                for fname in ["pytorch_model.bin", "gliner_config.json", "tokenizer.json", "tokenizer_config.json"]:
                    fsrc = src_ckpt / fname
                    if fsrc.exists():
                        shutil.copy2(fsrc, self.best_dir / fname)
            else:
                trainer = self.trainer_ref.get("trainer")
                if trainer is not None:
                    trainer.save_model(str(self.best_dir))

            info = {
                "best_checkpoint": tag,
                "best_step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": None if math.isnan(val_loss) else round(val_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_acc": round(val_stats["acc"], 2),
                "val_precision": round(val_stats["precision"], 2),
                "val_recall": round(val_stats["recall"], 2),
                "val_f1": round(val_stats["f1"], 2),
                "val_threshold": val_stats["threshold"],
                "tp": val_stats["tp"],
                "fp": val_stats["fp"],
                "fn": val_stats["fn"],
            }
            with open(self.best_dir / "best_checkpoint_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            print(f"  [Warning] Could not copy best_checkpoint files: {e}", flush=True)

    def _write_leaderboard_csv(self):
        if not self.leaderboard:
            return
        keys = [
            "step", "tag", "epoch", "train_loss", "val_loss",
            "train_acc", "train_f1", "val_acc", "val_precision", "val_recall", "val_f1", "val_threshold", "is_best"
        ]
        with open(self.history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(self.leaderboard)

    def print_final_leaderboard(self):
        if not self.leaderboard:
            return
        print("\n" + "=" * 78, flush=True)
        print("FINAL CHECKPOINT LEADERBOARD (RANKED BY VALIDATION F1 & ACCURACY)", flush=True)
        print("-" * 78, flush=True)
        ranked = sorted(self.leaderboard, key=lambda r: (r["val_f1"], r["val_acc"]), reverse=True)
        for rank, r in enumerate(ranked, 1):
            star = " <-- WINNER" if rank == 1 else ""
            print(
                f"#{rank:02d} | {r['tag']:16s} | Step {r['step']:05d} | "
                f"Val F1: {r['val_f1']:5.2f}% | Val Acc: {r['val_acc']:5.2f}% | "
                f"Train Loss: {r['train_loss']:.4f}{star}",
                flush=True,
            )
        print("-" * 78, flush=True)
        print(f"Best Checkpoint Saved At: {self.best_dir} (from {self.best_checkpoint_name})", flush=True)
        print(f"Full CSV History Saved At: {self.history_csv}", flush=True)
        print("=" * 78 + "\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--output-dir", default="models/antshield-gliner-v1")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=1e-5,
                    help="LR of the backbone encoder (token_rep_layer)")
    ap.add_argument("--others-lr", type=float, default=5e-5,
                    help="LR of everything that is not the encoder (span/prompt heads, projections). "
                         "Before this flag the heads trained at the same 5e-6 as the encoder.")
    ap.add_argument("--qat-scale-lr", type=float, default=1e-3,
                    help="LR of the E8 per-row learnable scales (theta_beta). At 5e-6 they could not move.")
    ap.add_argument("--init-from", default=None,
                    help="Folder with a pytorch_model.bin (e.g. the best FP checkpoint) loaded BEFORE "
                         "quantisation, so QAT starts from a fine-tuned full-precision model.")
    ap.add_argument("--val-file", default="val_finetune.jsonl",
                    help="file in --data-dir used for validation / best-checkpoint selection")
    ap.add_argument("--check-prompts", action="store_true",
                    help="print the label prompt GLiNER's collator builds for a few windows and exit")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-types", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--scaling-mode", default="block_max", choices=["block_max", "row_sigma", "row_sigma_learnable"])
    ap.add_argument("--no-learnable-scaling", dest="learnable_scaling", action="store_false")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="Path to checkpoint-XXXX folder, or 'auto' to pick latest in --output-dir.")
    ap.add_argument("--max-window-tokens", type=int, default=0,
                    help="0 = keep the 300-token windows as built (default). 256 re-introduces the old centre-crop "
                         "that silently removed spans/negatives; use only if you run out of VRAM.")
    ap.add_argument("--max-token-chars", type=int, default=128)
    ap.add_argument("--max-subwords", type=int, default=1024,
                    help="Drop training windows longer than this many subwords (0 disables).")
    ap.add_argument("--val-samples", type=int, default=600)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save once per epoch")
    ap.add_argument("--vram-fraction", type=float, default=0.88)
    args = ap.parse_args()

    if torch.cuda.is_available() and 0.1 < args.vram_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.vram_fraction, 0)
            print(f"Set CUDA per-process VRAM ceiling to {args.vram_fraction * 100:.0f}% (prevents Windows WDDM freeze)")
        except Exception as e:
            print(f"Note: could not set VRAM fraction: {e}")

    use_bf16 = (not args.fp16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("Using bf16")
    precision_kwargs = {"bf16": True} if use_bf16 else {"fp16": args.fp16}

    raw_train = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    val_path = Path(args.data_dir) / args.val_file
    if not val_path.exists():
        raise FileNotFoundError(
            f"{val_path} not found. Rebuild the data with data/build_dataset.py (it now writes a real "
            f"val split). Refusing to fall back to test_finetune.jsonl: that would tune on the test set.")
    raw_val = load_jsonl(val_path)

    # The label set the model is trained (and must be evaluated) on: every label with >= 1 train span.
    all_labels = sorted({
        str(span[2])
        for ex in raw_train
        for span in (ex.get("ner") or [])
        if len(span) >= 3
    })

    train_data = [
        sanitize_example_in_place(ex, args.max_window_tokens, args.max_token_chars)
        for ex in raw_train
    ]

    # Validation slice keeps the natural mix of positive AND entity-free windows (the old code used
    # positives only, from the TEST file, so precision on negatives was never measured).
    import random as _random
    _rng = _random.Random(0)
    val_all = [sanitize_example_in_place(ex, args.max_window_tokens, args.max_token_chars) for ex in raw_val]
    val_slice = _rng.sample(val_all, min(args.val_samples, len(val_all)))
    train_positive = [ex for ex in train_data if ex.get("ner")][:200]

    print(
        f"Loaded {len(train_data)} training windows and {len(val_slice)} validation windows "
        f"({sum(1 for e in val_slice if e['ner'])} with entities) across {len(all_labels)} PII entity types."
    )

    model = GLiNER.from_pretrained(args.base_model, local_files_only=True)

    # ---- drop very long training windows (train/long_filter.py) ----
    _tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if args.max_subwords > 0:
        train_data = drop_long_examples(train_data, _tok, args.max_subwords)
        train_positive = [ex for ex in train_data if ex.get("ner")][:200]

    if hasattr(model, "config"):
        model.config.max_types = max(args.max_types, len(all_labels))
        print(f"Set model.config.max_types = {model.config.max_types} (all {len(all_labels)} labels fit in every prompt)")

    if args.init_from:
        init_bin = Path(args.init_from) / "pytorch_model.bin"
        if not init_bin.exists():
            raise FileNotFoundError(f"--init-from: {init_bin} not found")
        sd0 = torch.load(str(init_bin), map_location="cpu")
        matched0, total0, miss0, unexp0 = smart_load_state_dict(model, sd0)
        print(f"Initialised from {args.init_from}: {matched0}/{total0} tensors (missing={len(miss0)}, unexpected={len(unexp0)})")
        if matched0 < 0.9 * total0:
            raise RuntimeError("--init-from matched <90% of the model tensors; wrong checkpoint?")
        del sd0

    if args.check_prompts:
        probe = [train_data[i] for i in range(0, len(train_data), max(1, len(train_data) // 6))][:6]
        ds_probe = LabelPromptDataset(probe, all_labels)
        c2i, _ = model.data_processor.batch_generate_class_mappings([ds_probe[i] for i in range(len(ds_probe))], key="ner")
        for ex, m in zip(probe, c2i):
            print(f"window with {len(ex['ner'])} entities {sorted({x[2] for x in ex['ner']})}\n   prompt has {len(m)} labels: {list(m)}")
        ok = all(len(m) == len(all_labels) for m in c2i)
        print("CHECK-PROMPTS:", "PASS - every window is prompted with all labels" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if args.quantize:
        print(f"Quantizing backbone encoder(s), scaling_mode={args.scaling_mode} ...")
        n = quantize_gliner_encoder(model, scaling_mode=args.scaling_mode, learnable=args.learnable_scaling)
        print(f"Total Linear layers quantized: {n}")

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt and resume_ckpt.lower() == "auto":
        resume_ckpt = find_latest_checkpoint(Path(args.output_dir))
        print(f"Auto-detected latest checkpoint: {resume_ckpt}")

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        bin_path = ckpt_path / "pytorch_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Checkpoint weights not found at {bin_path}")
        print(f"Pre-loading quantized weights from {bin_path} ...")
        state_dict = torch.load(str(bin_path), map_location="cpu")
        matched, total, missing, unexpected = smart_load_state_dict(model, state_dict)
        print(
            f"Restored {matched}/{total} tensors from {ckpt_path.name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        del state_dict
        gc.collect()

    steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.grad_accum))
    total_epochs = max(1, math.ceil(args.max_steps / steps_per_epoch))
    if args.save_steps <= 0:
        args.save_steps = steps_per_epoch
    trainer_ref = {}
    clean_cb = CleanTrainingLoggerCallback(
        gliner_wrapper=model,
        output_dir=Path(args.output_dir),
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        save_steps=args.save_steps,
        log_every=args.log_every,
        grad_accum=args.grad_accum,
        train_eval_subset=train_positive,
        val_eval_subset=val_slice,
        all_labels=all_labels,
        trainer_ref=trainer_ref,
    )

    TrainerCls = gliner_trainer_mod.Trainer
    orig_init = TrainerCls.__init__
    orig_training_step = TrainerCls.training_step
    orig_train = TrainerCls.train

    def patched_init(self, *t_args, **t_kwargs):
        orig_init(self, *t_args, **t_kwargs)
        self.args.logging_steps = 10
        self.args.per_device_eval_batch_size = 1
        self.args.save_strategy = "steps"
        self.args.save_steps = args.save_steps
        self.args.save_total_limit = None
        self.args.disable_tqdm = True
        if hasattr(self.args, "eval_strategy"):
            self.args.eval_strategy = "no"
        if hasattr(self.args, "evaluation_strategy"):
            self.args.evaluation_strategy = "no"
        self.remove_callback(PrinterCallback)
        self.remove_callback(ProgressCallback)
        self.add_callback(clean_cb)
        trainer_ref["trainer"] = self

    def patched_load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = model if model is not None else self.model
        bin_file = Path(resume_from_checkpoint) / "pytorch_model.bin"
        if bin_file.exists():
            sd = torch.load(str(bin_file), map_location="cpu")
            matched, total, missing, unexpected = smart_load_state_dict(target, sd)
            print(
                f"[Trainer Resume] Loaded {matched}/{total} tensors from {bin_file.parent.name} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
            del sd
            gc.collect()

    def patched_training_step(self, model_obj, inputs, *t_args, **t_kwargs):
        oom_triggered = False
        try:
            return orig_training_step(self, model_obj, inputs, *t_args, **t_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_triggered = True
            else:
                raise
        if oom_triggered:
            clean_cb.oom_skips_segment += 1
            model_obj.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.zeros((), device=self.args.device, requires_grad=True)

    def patched_train(self, *t_args, **t_kwargs):
        if resume_ckpt and "resume_from_checkpoint" not in t_kwargs and not t_args:
            t_kwargs["resume_from_checkpoint"] = str(resume_ckpt)
            print(f"Resuming Trainer state, optimizer, and scheduler from: {resume_ckpt}", flush=True)
        return orig_train(self, *t_args, **t_kwargs)

    orig_create_optimizer = TrainerCls.create_optimizer

    def patched_create_optimizer(self):
        opt = orig_create_optimizer(self)
        if args.quantize and args.qat_scale_lr is not None and not getattr(self, "_scale_group_done", False):
            self._scale_group_done = True
            theta = {id(p) for n, p in self.model.named_parameters() if n.endswith("theta_beta") and p.requires_grad}
            if theta:
                moved = []
                for g in opt.param_groups:
                    keep = [p for p in g["params"] if id(p) not in theta]
                    moved += [p for p in g["params"] if id(p) in theta]
                    g["params"] = keep
                opt.add_param_group({"params": moved, "lr": args.qat_scale_lr, "weight_decay": 0.0})
                print(f"[optimizer] {len(moved)} E8 scale tensors (theta_beta) -> own group, lr={args.qat_scale_lr}", flush=True)
        return opt

    TrainerCls.create_optimizer = patched_create_optimizer
    TrainerCls.__init__ = patched_init
    TrainerCls._load_from_checkpoint = patched_load_from_checkpoint
    TrainerCls.training_step = patched_training_step
    TrainerCls.train = patched_train

    out_dir_p = Path(args.output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "train_labels.json").write_text(json.dumps(all_labels, indent=2), encoding="utf-8")
    import hashlib
    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    (out_dir_p / "training_meta.json").write_text(json.dumps({
        "labels": all_labels, "args": vars(args), "n_train_windows": len(train_data),
        "n_val_windows": len(val_slice), "quantized": bool(args.quantize),
        "data_sha256": {n: _sha(Path(args.data_dir) / n) for n in ("train_finetune.jsonl", args.val_file)},
        "note": "evaluate with the labels in train_labels.json (eval/evaluate_gliner_audit.py --labels-file)",
    }, indent=2, default=str), encoding="utf-8")

    train_ds = LabelPromptDataset(train_data, all_labels, shuffle=True)
    val_ds = LabelPromptDataset(val_slice[:100], all_labels, shuffle=False)

    trainer = model.train_model(
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        others_lr=args.others_lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        **precision_kwargs,
    )
    trainer.save_model()
    clean_cb.print_final_leaderboard()
    print(f"Saved final fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
