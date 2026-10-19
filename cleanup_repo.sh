#!/usr/bin/env bash
# cleanup_repo.sh -- dry run by default; pass --apply to move files to the archive
ROOT="/mnt/c/Users/hp/Desktop/vishwajeet"
ARCH="$HOME/antshield_archive"
APPLY=0; [ "$1" = "--apply" ] && APPLY=1

# the audit project's venv is inside data_audit/, so it must not be active
case "${VIRTUAL_ENV:-}" in
  *data_audit*) echo "Run 'deactivate' first (the audit .venv is active)."; exit 1;;
esac
cd "$ROOT" || exit 1

TARGETS=(
  # earlier data insertion / filtering outputs
  data/augment_dataset.py data/augment_dataset_v2.py
  data/train_finetune.jsonl data/val_finetune.jsonl data/test_finetune.jsonl data/test_gliner.jsonl
  data/train_finetune.jsonl.bak data/val_finetune.jsonl.bak
  data_nl
  # audits
  data_audit
  # old probe / eval scripts that depend on val_finetune.jsonl, and their results
  evaluate_all.py evaluate_baseline.py eval_grouped.py eval_model.py fp_probe.py probe_missed.py sweep_checkpoints.py
  eval_grouped.txt eval_nl_t0.3.txt eval_nl_t0.4.txt eval_nl_test.txt fp_probe_results.txt sweep_results.txt
  # archives, junk
  antshield_public_dataset.zip gliner_actual_data_audit.zip gliner_data_pipeline.zip gliner_training_pipeline.zip
  "antshield-gliner (2).tar.gz" train/antshield_public_dataset.zip train/gliner_dataset_audit_preprocessing.zip
  __MACOSX __pycache__ data/__pycache__ eval/__pycache__ quantization/__pycache__
)

echo "Mode: $([ $APPLY = 1 ] && echo APPLY || echo DRY-RUN)   archive -> $ARCH"
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then
    printf '%-60s %s\n' "$t" "$(du -sh "$t" 2>/dev/null | cut -f1)"
    if [ $APPLY = 1 ]; then mkdir -p "$ARCH/$(dirname "$t")"; mv "$t" "$ARCH/$t"; fi
  fi
done

# restore the deleted training script (the deletion is uncommitted; HEAD still has it)
if [ ! -f train/finetune_gliner.py ]; then
  echo "restore: train/finetune_gliner.py (from HEAD)"
  [ $APPLY = 1 ] && git restore train/finetune_gliner.py
fi

echo; echo "--- git status after ---"; git status --short | head -40
[ $APPLY = 0 ] && echo && echo "Dry run only. Re-run with: bash ~/cleanup_repo.sh --apply"
