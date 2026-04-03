#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import csv
import re
import traceback
import sys
from random import shuffle, seed

# Your OCR/LLM function
from few_shot_ocr import transcribe_historical_document

# -------- CONFIG --------
IMAGES_DIR = Path("benchmark-images")   # folder of images
SAMPLE_LIMIT = 0  # set to 0 or None for full run
MODELS: List[str] = [
    "google/gemma-4-31b-it",
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3.5-flash-02-23",
    # "anthropic/claude-opus-4.6",
    # "google/gemini-2.5-flash","google/gemini-2.5-flash-lite",
    # "google/gemma-3-12b-it","google/gemma-3-27b-it",
    # "mistralai/pixtral-large-2411", "mistralai/pixtral-12b",
    # "amazon/nova-pro-v1",
    # "anthropic/claude-3.5-haiku", "anthropic/claude-sonnet-4",
    # "microsoft/phi-4-multimodal-instruct",
    # "openai/gpt-4.1-mini",  "openai/gpt-5-mini","openai/gpt-5-nano", "openai/gpt-5",
    # "meta-llama/llama-4-maverick","meta-llama/llama-4-scout",
    # "qwen/qwen2.5-vl-32b-instruct", "qwen/qwen2.5-vl-72b-instruct", "qwen/qwen-vl-plus",
    # "qwen/qwq-32b",
    # "mistralai/mistral-medium-3.1",
    # "mistralai/mistral-large-2512",
    # "google/gemini-3-pro-preview",
    # "openrouter/sonoma-dusk-alpha", "openrouter/sonoma-sky-alpha",
    # "meta-llama/llama-3.2-90b-vision-instruct",
    # "x-ai/grok-4-fast:free",
    # "qwen/qwen3-vl-235b-a22b-instruct",
    # "google/gemini-2.5-flash-lite-preview-09-2025",
    # "google/gemini-2.5-flash-preview-09-2025",
    # "anthropic/claude-haiku-4.5",
    # "qwen/qwen3-vl-8b-thinking",
    # "qwen/qwen3-vl-8b-instruct",
    # "qwen/qwen3-vl-32b-instruct",
    # "nvidia/nemotron-nano-12b-v2-vl",
    # "nvidia/nemotron-nano-12b-v2-vl:free"
    # "opengvlab/internvl3-78b" exclude becuase of 32K context window is too small.
    # add more models here
    ]

#MODELS: List[str] = [
#    "google/gemini-2.5-flash-lite",
#    "openai/gpt-5-mini", "openai/gpt-5-nano", ]

shuffle(MODELS)
MAX_WORKERS = 4        # how many calls to run in parallel
OUTPUT_ROOT = Path("ocr-results")
# ------------------------

def sanitize_for_path(name: str) -> str:
    """Make a safe folder name from a model id (strip slashes, spaces, etc.)."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")

def output_file_for(image_path: Path, model: str) -> Path:
    """Save as ocr-results/<model>/<image_stem>.txt"""
    model_dir = OUTPUT_ROOT / sanitize_for_path(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{image_path.stem}.txt"

USAGE_LOG = OUTPUT_ROOT / "token_usage.csv"

def ensure_usage_log():
    """Create the usage CSV with headers if it doesn't exist."""
    if not USAGE_LOG.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "model", "prompt_tokens", "completion_tokens", "total_tokens"])

def log_usage(image_name: str, model: str, usage: dict):
    """Append a row to the token usage log."""
    with open(USAGE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_name,
            model,
            usage.get("prompt_tokens", ""),
            usage.get("completion_tokens", ""),
            usage.get("total_tokens", ""),
        ])

def transcribe_once(image_path: Path, model: str) -> Path:
    """Run a transcription unless already present with content."""
    out_path = output_file_for(image_path, model)

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path  # already done

    text, usage = transcribe_historical_document(str(image_path), model=model, return_usage=True)
    out_path.write_text(text if text else "", encoding="utf-8")
    if usage:
        log_usage(image_path.name, model, usage)
    return out_path

def main():
    if not IMAGES_DIR.exists():
        print(f"ERROR: folder not found: {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".tiff"})
    seed(42)
    shuffle(images)
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    if SAMPLE_LIMIT:
        images = images[:SAMPLE_LIMIT]
        print(f"SAMPLE MODE: processing {SAMPLE_LIMIT} images only")

    ensure_usage_log()

    def run_model(model, images):
        """Run all images for a single model sequentially."""
        for img in images:
            out_path = output_file_for(img, model)
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[SKIP] {img.name} ({model})")
                continue
            try:
                out_path = transcribe_once(img, model)
                print(f"[OK]   {img.name} ({model}) -> {out_path}")
            except Exception:
                print(f"[FAIL] {img.name} ({model})", file=sys.stderr)
                traceback.print_exc()

    # Run all models concurrently, each processing images sequentially
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {pool.submit(run_model, model, images): model for model in MODELS}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                fut.result()
                print(f"\n[DONE] {model} finished all images.")
            except Exception:
                print(f"\n[ERROR] {model} crashed.", file=sys.stderr)
                traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
