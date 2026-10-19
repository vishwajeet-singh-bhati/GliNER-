#!/usr/bin/env bash
# phase1b_inventory.sh -- READ-ONLY. Writes only to ~/phase1b_report.txt
ROOT="/mnt/c/Users/hp/Desktop/vishwajeet"
AUD="$ROOT/data_audit/gliner_dataset_audit_preprocessing/gliNER-dataset-audit"
OUT="$HOME/phase1b_report.txt"
exec > >(tee "$OUT") 2>&1
section() { printf '\n==== %s ====\n' "$1"; }
cd "$ROOT" || exit 1

section "A1. .gitignore"; cat .gitignore
section "A2. Tracked vs ignored vs untracked"
for p in data/train_finetune.jsonl data/val_finetune.jsonl data/test_finetune.jsonl data/test_gliner.jsonl \
         data/train_finetune.jsonl.bak data/val_finetune.jsonl.bak data_nl/train_finetune.jsonl \
         configs/productive_header_keys.json models "antshield-gliner (2).tar.gz" gliner_data_pipeline.zip \
         train/antshield_public_dataset.zip antshield_public_dataset; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then s=TRACKED
  elif git check-ignore -q "$p"; then s=IGNORED
  else s=UNTRACKED; fi
  printf '%-10s %s\n' "$s" "$p"
done
section "A3. HEAD commit"; git show --stat --format='%h %ad %s%n%b' --date=short HEAD | head -40
section "A4. Uncommitted diffs"
git diff --stat
git diff -- data/filter_by_subwords.py eval/zero_shot_eval.py | head -250
section "A5. Deleted training script (from HEAD)"
git log --oneline -- train/finetune_gliner.py | head
git show HEAD:train/finetune_gliner.py | wc -l
git show HEAD:train/finetune_gliner.py | head -40
git show HEAD:train/finetune_gliner.py | grep -nE 'tokenized_text|ner_labels|"ner"|Trainer|train_model|data_nl|train_finetune|max_len|max_width|from_pretrained' | cut -c1-200

section "B1. Real files (excluding __MACOSX, raw_data, venvs, checkpoint dirs)"
find "$ROOT" \( -name .venv -o -name venv -o -name .git -o -name __pycache__ -o -name node_modules \
  -o -name __MACOSX -o -name raw_data -o -path '*/models/*/checkpoint-*' \) -prune -o \
  -type f \( -name '*.jsonl' -o -name '*.json' -o -name '*.zip' -o -name '*.tar*' -o -name '*.csv' \
  -o -name '*.txt' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.cfg' \
  -o -name '*.bak' -o -name '*.pt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.onnx' \) \
  -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | sort -k3
section "B2. Model dirs and checkpoint sizes"
du -sh models/* models/*/checkpoint-* 2>/dev/null
section "B3. Top-level disk use"
du -sh "$ROOT"/* 2>/dev/null | sort -h | tail -15
df -h "$ROOT" "$HOME" | cat

section "C. Archive contents (listing only, nothing extracted)"
for z in antshield_public_dataset.zip gliner_actual_data_audit.zip gliner_data_pipeline.zip \
         gliner_training_pipeline.zip train/antshield_public_dataset.zip train/gliner_dataset_audit_preprocessing.zip; do
  echo "--- $z"; ls -l "$z"
  python3 -m zipfile -l "$z" 2>&1 | grep -vE '__MACOSX|raw_data/|\._' | head -60
done
t="antshield-gliner (2).tar.gz"; echo "--- $t"; ls -l "$t"
if [ "$(stat -c %s "$t")" -lt 2147483648 ]; then
  timeout 120 tar tzf "$t" 2>&1 | grep -vE '__MACOSX|raw_data/|/\._' | head -80
else echo "skipped (>2 GiB)"; fi

section "D. Raw dataset copies (sizes, counts, content hash of relative paths)"
for d in "$ROOT/antshield_public_dataset" \
         "$ROOT/data_audit/antshield_public_dataset/antshield_public_dataset" \
         "$AUD/data/original/antshield_public_dataset"; do
  echo "--- $d"
  [ -d "$d/raw_data" ] || { echo "no raw_data here"; continue; }
  du -sh "$d"
  echo "json files:       $(find "$d/raw_data" -type f -name '*.json' ! -name '._*' | wc -l)"
  echo "AppleDouble ._*:  $(find "$d/raw_data" -name '._*' | wc -l)"
  echo "auto_anteater:    $(find "$d/raw_data/auto_anteater" -type f -name '*.json' ! -name '._*' | wc -l)"
  echo "manual_anteater:  $(find "$d/raw_data/manual_anteater" -type f -name '*.json' ! -name '._*' | wc -l)"
  echo -n "content hash:     "
  find "$d/raw_data" -type f -name '*.json' ! -name '._*' -print0 | sort -z | xargs -0 sha256sum \
    | sed 's#  .*/raw_data/#  #' | sha256sum
  for c in apps_info.csv exposures.csv; do [ -f "$d/$c" ] && sha256sum "$d/$c"; done
done

section "E. jsonl key-sets per line (no values printed) + .bak hashes"
sha256sum data/*.bak; wc -l data/*.bak
python3 - <<'PY'
import json, collections, pathlib
root = pathlib.Path("/mnt/c/Users/hp/Desktop/vishwajeet")
files = ["data/train_finetune.jsonl","data/val_finetune.jsonl","data/test_finetune.jsonl","data/test_gliner.jsonl",
         "data/train_finetune.jsonl.bak","data/val_finetune.jsonl.bak",
         "data_nl/train_finetune.jsonl","data_nl/val_finetune.jsonl"]
for f in files:
    p = root / f
    if not p.exists(): print(f, "MISSING"); continue
    c = collections.Counter(); bad = n = 0; first = None
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            n += 1
            try: o = json.loads(line)
            except Exception: bad += 1; continue
            if first is None and isinstance(o, dict):
                first = {k: (type(v).__name__ + (f"[{len(v)}]" if hasattr(v, "__len__") else "")) for k, v in o.items()}
            c[tuple(sorted(o)) if isinstance(o, dict) else ("<%s>" % type(o).__name__,)] += 1
    print(f"{f}: lines={n} unparseable={bad}")
    for k, v in c.most_common(5): print("   ", v, k)
    print("    first record shape:", first)
PY

section "F. README, requirements, configs"
cat README.md
echo "--- requirements.txt"; cat requirements.txt
echo "--- audit requirements.txt"; cat "$AUD/requirements.txt"
python3 -c "import json;d=json.load(open('configs/productive_header_keys.json'));print('productive_header_keys:',type(d).__name__,len(d))"
echo "--- mobile/NOTES.md (head)"; head -30 mobile/NOTES.md

section "G. Other Python environments"
find "$ROOT" -maxdepth 5 -name pyvenv.cfg -not -path '*/__MACOSX/*' 2>/dev/null
find "$HOME" -maxdepth 3 -name pyvenv.cfg 2>/dev/null
for py in /usr/bin/python3 "$(which python3)"; do
  echo "--- $py"
  $py -c "import importlib.metadata as m;print(sorted({d.metadata['Name'] for d in m.distributions()}))"
done

section "H. Code-level entry/CLI facts (grep only)"
grep -rn --include='*.py' 'sys.path' data eval quantization *.py 2>/dev/null | cut -c1-160
echo "--- argparse in pipeline scripts"
grep -n -A1 'add_argument' data/prepare_finetune_data.py data/convert_to_gliner_format.py \
  data/filter_by_subwords.py data/antshield_loader.py data/analyze_pii_bearing_fields.py 2>/dev/null | cut -c1-200
echo "--- third-party imports"
grep -hnE '^(import|from) ' data/*.py eval/*.py 2>/dev/null \
  | grep -vE 'from (pii_type_mapping|antshield_loader|recon_synthesizer|splitting|gliner_tokenizer|chunking)' | sort -u | cut -c1-120

section "I. Audit project: layout, summary, workflow outputs"
ls "$AUD/config" "$AUD/docs" "$AUD/logs" "$AUD/data" 2>&1
cat "$AUD/README.md"
cat "$AUD/reports/real/AUDIT_SUMMARY.md"
cat "$AUD/scripts/run_workflow.py"

echo; echo "Saved to $OUT"
