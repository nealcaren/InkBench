#!/usr/bin/env python3
import csv
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics as stats

# ---------- CONFIG ----------
IMAGES_DIR = Path("benchmark-images")
REF_DIR = Path("benchmark-txt")
HYPS_ROOT = Path("ocr-results")
BENCHMARK_CSV = Path("/Users/nealcaren/Downloads/InkBench/benchmark.csv")

RESULTS_CSV = Path("ocr_eval_results.csv")
SUMMARY_BY_MODEL_CSV = Path("ocr_eval_summary_by_model.csv")
SUMMARY_BY_MODEL_AND_TYPE_CSV = Path("ocr_eval_summary_by_model_and_type.csv")
MODEL_ACCURACY_WIDE_CSV = Path("ocr_eval_model_accuracy.csv")  # <<< NEW

# jiwer normalization knobs
LOWERCASE = True
NORMALIZE_SPACES = True
STRIP = True
# ----------------------------

# Nice labels for the benchmark "type" column
TYPE_LABELS = {
    "BOOK_PAGE": "Book Page",
    "HANDWRITTEN": "Handwritten",
    "MIXED": "Mixed",
    "OTHER_TYPED_OR_PRINTED": "Other Typed/Printed",
}
TYPE_ORDER = ["BOOK_PAGE", "HANDWRITTEN", "MIXED", "OTHER_TYPED_OR_PRINTED"]

# ==== JIWER v4 ====
import jiwer
from jiwer import (
    Compose,
    ToLowerCase,
    RemoveMultipleSpaces,
    Strip,
    ReduceToListOfListOfWords,
    ReduceToListOfListOfChars,
    process_words,
    visualize_alignment,
    collect_error_counts,
)

def make_words_transform():
    steps = []
    if LOWERCASE:
        steps.append(ToLowerCase())
    if NORMALIZE_SPACES:
        steps.append(RemoveMultipleSpaces())
    if STRIP:
        steps.append(Strip())
    steps.append(ReduceToListOfListOfWords())
    return Compose(steps)

def make_chars_transform():
    steps = []
    if STRIP:
        steps.append(Strip())
    if LOWERCASE:
        steps.append(ToLowerCase())
    steps.append(ReduceToListOfListOfChars())
    return Compose(steps)

WORDS_TRANSFORM = make_words_transform()
CHARS_TRANSFORM = make_chars_transform()
ALNUM_CHARS_ONLY = Compose([ReduceToListOfListOfChars()])  # used after manual filtering

# --- helpers for alnum+lower CER ---
_ALNUM_RE = re.compile(r'[^0-9a-z]')
def _alnum_lower(s: str) -> str:
    # lower() then strip everything not [0-9a-z]
    return _ALNUM_RE.sub('', s.lower())

def pair_measures(ref: str, hyp: str) -> Tuple[float, float, float]:
    """Return (WER, CER, CER_ALNUM)."""
    w = jiwer.wer(
        ref, hyp,
        reference_transform=WORDS_TRANSFORM,
        hypothesis_transform=WORDS_TRANSFORM,
    )
    c = jiwer.cer(
        ref, hyp,
        reference_transform=CHARS_TRANSFORM,
        hypothesis_transform=CHARS_TRANSFORM,
    )
    # Alphanumeric+lower cer: pre-filter, then score with bare char reducer
    ref_a = _alnum_lower(ref)
    hyp_a = _alnum_lower(hyp)
    c_alnum = jiwer.cer(
        ref_a, hyp_a,
        reference_transform=ALNUM_CHARS_ONLY,
        hypothesis_transform=ALNUM_CHARS_ONLY,
    )
    return float(w), float(c), float(c_alnum)

def read_text(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return None

def iter_models(hyps_root: Path) -> List[str]:
    return sorted([d.name for d in hyps_root.iterdir() if d.is_dir()])

def load_benchmark_list(csv_path: Path):
    assert csv_path.exists(), f"Missing benchmark CSV: {csv_path}"
    entries: List[Dict[str, str]] = []
    chosen: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"image_name", "type", "source_dir"}
        missing_cols = required - set(r.fieldnames or [])
        assert not missing_cols, f"benchmark.csv missing columns: {missing_cols}"
        for row in r:
            img = row["image_name"].strip()
            typ = row["type"].strip()
            src = row["source_dir"].strip()
            if img and img not in chosen:  # keep first occurrence
                chosen[img] = {"type": typ, "source_dir": src}
            entries.append({"image_name": img, "type": typ, "source_dir": src})
    return entries, chosen

def _agg(values: List[float], mode: str) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    if mode == "median":
        return float(stats.median(vals))
    return float(sum(vals) / len(vals))  # mean

def evaluate(align_n: int = 0,
             dump_error_counts_path: Optional[Path] = None,
             metric_for_accuracy: str = "cer_alnum",  # wer|cer|cer_alnum
             agg_for_accuracy: str = "mean"           # mean|median
             ):
    assert REF_DIR.exists(), f"Missing folder: {REF_DIR}"
    assert HYPS_ROOT.exists(), f"Missing folder: {HYPS_ROOT}"

    metric_for_accuracy = metric_for_accuracy.lower()
    agg_for_accuracy = agg_for_accuracy.lower()
    assert metric_for_accuracy in {"wer", "cer", "cer_alnum"}
    assert agg_for_accuracy in {"mean", "median"}

    models = iter_models(HYPS_ROOT)
    if not models:
        print("No model folders found in ocr-results.")
        return

    bench_entries, _ = load_benchmark_list(BENCHMARK_CSV)

    fieldnames = [
        "image_name", "type", "source_dir", "model", "status",
        "wer", "cer", "cer_alnum", "ref_word_count", "hyp_word_count"
    ]

    rows = []

    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_model_type: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    refs_by_model: Dict[str, List[str]] = defaultdict(list)
    hyps_by_model: Dict[str, List[str]] = defaultdict(list)
    failures_for_align: List[Tuple[str, str, str, str]] = []

    for entry in bench_entries:
        image_name = entry["image_name"]
        typ = entry["type"]
        src = entry["source_dir"]
        stem = Path(image_name).stem

        ref_path = REF_DIR / f"{stem}.txt"
        ref_text = read_text(ref_path)

        for m in models:
            hyp_path = HYPS_ROOT / m / f"{stem}.txt"
            hyp_text = read_text(hyp_path)

            if ref_text is None:
                rows.append({
                    "image_name": image_name, "type": typ, "source_dir": src,
                    "model": m, "status": "missing_ref", "wer": "", "cer": "", "cer_alnum": "",
                    "ref_word_count": "", "hyp_word_count": ""
                })
                continue
            if hyp_text is None:
                rows.append({
                    "image_name": image_name, "type": typ, "source_dir": src,
                    "model": m, "status": "missing_hyp", "wer": "", "cer": "", "cer_alnum": "",
                    "ref_word_count": "", "hyp_word_count": ""
                })
                continue

            w, c, c_alnum = pair_measures(ref_text, hyp_text)

            # Count words (after transforms)
            ref_wc = len(WORDS_TRANSFORM(ref_text)[0]) if ref_text else 0
            hyp_wc = len(WORDS_TRANSFORM(hyp_text)[0]) if hyp_text else 0

            rows.append({
                "image_name": image_name, "type": typ, "source_dir": src,
                "model": m, "status": "ok",
                "wer": w, "cer": c, "cer_alnum": c_alnum,
                "ref_word_count": ref_wc, "hyp_word_count": hyp_wc
            })

            per_model[m]["wer"].append(w)
            per_model[m]["cer"].append(c)
            per_model[m]["cer_alnum"].append(c_alnum)

            per_model_type[m][typ]["wer"].append(w)
            per_model_type[m][typ]["cer"].append(c)
            per_model_type[m][typ]["cer_alnum"].append(c_alnum)

            refs_by_model[m].append(ref_text)
            hyps_by_model[m].append(hyp_text)
            if align_n > 0 and (w > 0 or c > 0) and len(failures_for_align) < align_n:
                failures_for_align.append((m, image_name, ref_text, hyp_text))

    # Write detailed per-sample results
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Per-model summary
    with SUMMARY_BY_MODEL_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_ok", "wer_mean", "cer_mean", "cer_alnum_mean"])
        for m in models:
            ws = per_model[m]["wer"]
            cs = per_model[m]["cer"]
            cas = per_model[m]["cer_alnum"]
            num_ok = len(ws)
            w.writerow([
                m,
                num_ok,
                (sum(ws) / num_ok) if num_ok else "",
                (sum(cs) / num_ok) if num_ok else "",
                (sum(cas) / num_ok) if num_ok else "",
            ])

    # Per-model × type summary
    with SUMMARY_BY_MODEL_AND_TYPE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "type", "num_ok", "wer_mean", "cer_mean", "cer_alnum_mean"])
        for m in models:
            type_map = per_model_type[m]
            for typ, metrics in sorted(type_map.items()):
                ws = metrics["wer"]
                cs = metrics["cer"]
                cas = metrics["cer_alnum"]
                num_ok = len(ws)
                w.writerow([
                    m, typ, num_ok,
                    (sum(ws) / num_ok) if num_ok else "",
                    (sum(cs) / num_ok) if num_ok else "",
                    (sum(cas) / num_ok) if num_ok else "",
                ])

    # ===== NEW: wide accuracy table (model × [Overall + per-type]) =====
    metric_key = metric_for_accuracy  # 'wer' | 'cer' | 'cer_alnum'

    # Build rows with accuracy = 1 - aggregate(error)
    wide_rows = []
    for m in models:
        overall_err = _agg(per_model[m][metric_key], agg_for_accuracy)
        overall_acc = (1.0 - overall_err) if overall_err is not None else None

        row = {
            "model": m,
            "Overall": overall_acc
        }
        for typ_code in TYPE_ORDER:
            err_list = per_model_type[m][typ_code][metric_key]
            typ_err = _agg(err_list, agg_for_accuracy)
            row[TYPE_LABELS[typ_code]] = (1.0 - typ_err) if typ_err is not None else None

        wide_rows.append(row)

    # Sort best-first by Overall accuracy (descending), None goes last
    def _sort_key(r):
        return (-r["Overall"], r["model"]) if r["Overall"] is not None else (1e9, r["model"])
    wide_rows.sort(key=_sort_key)

    # Write the wide CSV
    headers = ["model", "Overall"] + [TYPE_LABELS[t] for t in TYPE_ORDER]
    with MODEL_ACCURACY_WIDE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in wide_rows:
            # Optional: round to 4 decimals for readability
            out = {}
            for k in headers:
                v = r.get(k, None)
                out[k] = (round(v, 4) if isinstance(v, float) else ("" if v is None else v))
            w.writerow(out)

    print(f"Wrote detailed results to: {RESULTS_CSV}")
    print(f"Wrote per-model summary to: {SUMMARY_BY_MODEL_CSV}")
    print(f"Wrote per-model-by-type summary to: {SUMMARY_BY_MODEL_AND_TYPE_CSV}")
    print(f"Wrote model accuracy table to: {MODEL_ACCURACY_WIDE_CSV} "
          f"(metric={metric_for_accuracy}, agg={agg_for_accuracy}; accuracy = 1 - agg(metric))")

    if align_n > 0 and failures_for_align:
        print("\n=== SAMPLE ALIGNMENTS (jiwer.visualize_alignment) ===")
        for (m, img, ref_text, hyp_text) in failures_for_align:
            out = process_words(
                [ref_text], [hyp_text],
                reference_transform=WORDS_TRANSFORM,
                hypothesis_transform=WORDS_TRANSFORM,
            )
            print(f"\n--- Model: {m} | Image: {img} ---")
            print(visualize_alignment(out, show_measures=True))

    if dump_error_counts_path:
        with dump_error_counts_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "error_type", "from", "to", "count"])
            for m in models:
                if not refs_by_model[m]:
                    continue
                out = process_words(
                    refs_by_model[m], hyps_by_model[m],
                    reference_transform=WORDS_TRANSFORM,
                    hypothesis_transform=WORDS_TRANSFORM,
                )
                subs, ins, dels = collect_error_counts(out)
                for (frm, to), cnt in sorted(subs.items(), key=lambda kv: (-kv[1], kv[0])):
                    w.writerow([m, "substitution", frm, to, cnt])
                for token, cnt in sorted(ins.items(), key=lambda kv: (-kv[1], kv[0])):
                    w.writerow([m, "insertion", "", token, cnt])
                for token, cnt in sorted(dels.items(), key=lambda kv: (-kv[1], kv[0])):
                    w.writerow([m, "deletion", token, "", cnt])

        print(f"Wrote model-level error counts to: {dump_error_counts_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR outputs with jiwer v4 and produce accuracy tables."
    )
    parser.add_argument("--align", type=int, default=0,
                        help="Print visualize_alignment for the first N non-OK pairs (default: 0).")
    parser.add_argument("--dump-error-counts", type=Path, default=None,
                        help="Optional path to write aggregated error counts CSV per model.")
    parser.add_argument("--metric", type=str, default="cer_alnum",
                        choices=["wer", "cer", "cer_alnum"],
                        help="Metric to invert into accuracy for the wide table (default: cer_alnum).")
    parser.add_argument("--agg", type=str, default="mean",
                        choices=["mean", "median"],
                        help="Aggregation for the wide table (default: mean).")
    args = parser.parse_args()
    evaluate(
        align_n=args.align,
        dump_error_counts_path=args.dump_error_counts,
        metric_for_accuracy=args.metric,
        agg_for_accuracy=args.agg
    )

if __name__ == "__main__":
    main()
