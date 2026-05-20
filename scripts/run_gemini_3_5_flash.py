#!/usr/bin/env python3
"""One-off driver to transcribe the InkBench corpus with google/gemini-3.5-flash."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import sys
import traceback

from few_shot_ocr import transcribe_historical_document

MODEL = "google/gemini-3.5-flash"
MODEL_SLUG = "google-gemini-3.5-flash"
IMAGES_DIR = Path("benchmark-images")
OUTPUT_ROOT = Path("ocr-results")
USAGE_LOG = OUTPUT_ROOT / "token_usage.csv"
MAX_WORKERS = 12


def ensure_usage_log():
    if not USAGE_LOG.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "model", "prompt_tokens", "completion_tokens", "total_tokens"])


def log_usage(image_name, usage):
    with open(USAGE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_name,
            MODEL,
            usage.get("prompt_tokens", ""),
            usage.get("completion_tokens", ""),
            usage.get("total_tokens", ""),
        ])


def transcribe(img: Path):
    out_dir = OUTPUT_ROOT / MODEL_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img.stem}.txt"
    if out_path.exists() and out_path.stat().st_size > 0:
        return img.name, "SKIP"
    try:
        text, usage = transcribe_historical_document(str(img), model=MODEL, return_usage=True)
        out_path.write_text(text if text else "", encoding="utf-8")
        if usage:
            log_usage(img.name, usage)
        return img.name, "OK"
    except Exception as e:
        traceback.print_exc()
        return img.name, f"FAIL: {e}"


def main():
    ensure_usage_log()
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    print(f"Transcribing {len(images)} images with {MODEL} (workers={MAX_WORKERS})", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(transcribe, img): img for img in images}
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{len(images)}] {status} {name}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
