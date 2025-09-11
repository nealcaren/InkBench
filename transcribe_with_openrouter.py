#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import re
import traceback
import sys
from random import shuffle

# Your OCR/LLM function
from few_shot_ocr import transcribe_historical_document

# -------- CONFIG --------
IMAGES_DIR = Path("benchmark-images")   # folder of images
MODELS: List[str] = [
    "google/gemini-2.5-flash","google/gemini-2.5-flash-lite",
    "google/gemma-3-12b-it","google/gemma-3-27b-it", 
    "mistralai/pixtral-large-2411", "mistralai/pixtral-12b",
    "amazon/nova-pro-v1",
    "anthropic/claude-3.5-haiku", "microsoft/phi-4-multimodal-instruct",
    "openai/gpt-4.1-mini",  "openai/gpt-5-mini","openai/gpt-5-nano",
    "meta-llama/llama-4-maverick","meta-llama/llama-4-scout",
    "qwen/qwen2.5-vl-32b-instruct", "qwen/qwen2.5-vl-72b-instruct", "qwen/qwen-vl-plus",
    "mistralai/mistral-medium-3.1"
    # add more models here
    ]
shuffle(MODELS)
MAX_WORKERS = 8        # how many calls to run in parallel
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

def transcribe_once(image_path: Path, model: str) -> Path:
    """Run a transcription unless already present with content."""
    out_path = output_file_for(image_path, model)

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path  # already done

    text = transcribe_historical_document(str(image_path), model=model)
    out_path.write_text(text if text else "", encoding="utf-8")
    return out_path

def main():
    if not IMAGES_DIR.exists():
        print(f"ERROR: folder not found: {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".tiff"})
    shuffle(images)
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for img in images:
            for model in MODELS:
                out_path = output_file_for(img, model)
                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"[SKIP] {img.name} ({model}) -> already exists")
                    continue
                fut = pool.submit(transcribe_once, img, model)
                tasks.append((img, model, fut))

        for img, model, fut in tasks:
            try:
                out_path = fut.result()
                print(f"[OK]   {img.name} ({model}) -> {out_path}")
            except Exception:
                print(f"[FAIL] {img.name} ({model})", file=sys.stderr)
                traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
