# Patches train\finetune_gliner.py to apply subword-aware length control.
# Backs up first, keeps line endings, aborts without writing if any anchor is missing.
$path = "train\finetune_gliner.py"
$full = (Resolve-Path $path).Path
$utf8 = New-Object System.Text.UTF8Encoding($false)
Copy-Item $full "$full.lenbak" -Force

$raw  = [IO.File]::ReadAllText($full, $utf8)
$crlf = $raw.Contains("`r`n")
$src  = $raw -replace "`r`n", "`n"

if ($src.Contains("fit_dataset")) { Write-Host "Already patched."; return }

$a1 = "from quantize_gliner import quantize_gliner_encoder`n"
$a2 = '    ap.add_argument("--val-samples", type=int, default=300)' + "`n"
$a3 = '    model = GLiNER.from_pretrained(args.base_model, local_files_only=True)' + "`n"

foreach ($a in @($a1, $a2, $a3)) {
    if (([regex]::Matches($src, [regex]::Escape($a))).Count -ne 1) {
        Write-Host "Anchor not found exactly once: $a  -> file NOT modified."
        return
    }
}

$imp = "from length_utils import fit_dataset, print_fit_report`n"

$arg = @'
    ap.add_argument("--max-subwords", type=int, default=384,
                    help="Max TEXT subwords per window (0 disables). See train/length_utils.py.")

'@ -replace "`r`n", "`n"

$blk = @'

    # ---- subword-aware length control (train/length_utils.py) ----
    _tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if args.max_subwords > 0:
        train_data, _st = fit_dataset(train_data, _tok, args.max_subwords)
        print_fit_report("train", _st)
        val_slice, _sv = fit_dataset(val_slice, _tok, args.max_subwords)
        print_fit_report("val", _sv)
        train_positive = [ex for ex in train_data if ex.get("ner")][:200]

'@ -replace "`r`n", "`n"

$src = $src.Replace($a1, $a1 + $imp)
$src = $src.Replace($a2, $arg + $a2)
$src = $src.Replace($a3, $a3 + $blk)

if ($crlf) { $src = $src -replace "`n", "`r`n" }
[IO.File]::WriteAllText($full, $src, $utf8)
Write-Host "Patched. Backup at $full.lenbak"