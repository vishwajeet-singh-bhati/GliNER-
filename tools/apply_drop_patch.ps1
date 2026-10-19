# Reverts finetune_gliner.py to its state before the trim patch (restoring
# finetune_gliner.py.lenbak, which still has the clear_inference_tensors fix),
# then adds ONLY a filter that drops over-long training windows.
$full = (Resolve-Path "train\finetune_gliner.py").Path
$bak  = "$full.lenbak"
$utf8 = New-Object System.Text.UTF8Encoding($false)

$cur = [IO.File]::ReadAllText($full, $utf8)
if ($cur.Contains("drop_long_examples") -and -not $cur.Contains("fit_dataset")) {
    Write-Host "Already patched."; return
}
if (-not (Test-Path $bak)) { Write-Host "Backup $bak not found. Nothing changed."; return }
$orig = [IO.File]::ReadAllText($bak, $utf8)
if ($orig.Contains("fit_dataset")) { Write-Host "Backup already contains the trim patch. Nothing changed."; return }
if (-not $orig.Contains("def clear_inference_tensors")) { Write-Host "WARNING: backup lacks clear_inference_tensors." }

$crlf = $orig.Contains("`r`n")
$src  = $orig -replace "`r`n", "`n"

$a1 = "from quantize_gliner import quantize_gliner_encoder`n"
$a2 = '    ap.add_argument("--val-samples", type=int, default=300)' + "`n"
$a3 = '    model = GLiNER.from_pretrained(args.base_model, local_files_only=True)' + "`n"
foreach ($a in @($a1, $a2, $a3)) {
    if (([regex]::Matches($src, [regex]::Escape($a))).Count -ne 1) {
        Write-Host "Anchor not found exactly once: $a -> nothing changed."; return
    }
}

$imp = "from long_filter import drop_long_examples`n"
$arg = @'
    ap.add_argument("--max-subwords", type=int, default=768,
                    help="Drop training windows longer than this many subwords (0 disables).")

'@ -replace "`r`n", "`n"
$blk = @'

    # ---- drop very long training windows (train/long_filter.py) ----
    _tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if args.max_subwords > 0:
        train_data = drop_long_examples(train_data, _tok, args.max_subwords)
        train_positive = [ex for ex in train_data if ex.get("ner")][:200]

'@ -replace "`r`n", "`n"

$src = $src.Replace($a1, $a1 + $imp)
$src = $src.Replace($a2, $arg + $a2)
$src = $src.Replace($a3, $a3 + $blk)
if ($crlf) { $src = $src -replace "`n", "`r`n" }

Copy-Item $full "$full.trimpatched" -Force
[IO.File]::WriteAllText($full, $src, $utf8)
Write-Host "Done. Trim version saved as $full.trimpatched"