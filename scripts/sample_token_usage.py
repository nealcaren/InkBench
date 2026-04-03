#!/usr/bin/env python3
"""
Sample 20 images across all models to get consistent token usage data for cost estimation.
Forces API calls even if transcription results already exist.
Writes to ocr-results/token_usage_sample.csv
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import seed, shuffle
import csv
import traceback
import sys

from few_shot_ocr import transcribe_historical_document

IMAGES_DIR = Path("benchmark-images")
SAMPLE_SIZE = 20
OUTPUT_CSV = Path("ocr-results/token_usage_sample.csv")

# All models that have results (excluding non-API ones like Tesseract/EasyOcr/finetuned)
MODELS = [
    "amazon/nova-pro-v1",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "google/gemini-2.5-flash-preview-09-2025",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/gemma-4-31b-it",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "microsoft/phi-4-multimodal-instruct",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-medium-3.1",
    "mistralai/pixtral-12b",
    "mistralai/pixtral-large-2411",
    "moonshotai/kimi-k2.5",
    "nvidia/nemotron-nano-12b-v2-vl",
    "openai/gpt-4.1-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openrouter/sonoma-dusk-alpha",
    "openrouter/sonoma-sky-alpha",
    "qwen/qwen-vl-plus",
    "qwen/qwen2.5-vl-32b-instruct",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-8b-thinking",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwq-32b",
    "x-ai/grok-4-fast:free",
]


def sample_model(model, images):
    """Run sample images for one model, return list of usage dicts."""
    results = []
    for img in images:
        try:
            text, usage = transcribe_historical_document(
                str(img), model=model, return_usage=True
            )
            if usage:
                results.append({
                    "image": img.name,
                    "model": model,
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                })
                print(f"[OK]   {img.name} ({model}) "
                      f"in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')}")
            else:
                print(f"[WARN] {img.name} ({model}) - no usage data")
        except Exception:
            print(f"[FAIL] {img.name} ({model})", file=sys.stderr)
            traceback.print_exc()
    return results


def main():
    images = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".tiff"}
    )
    seed(42)
    shuffle(images)
    images = images[:SAMPLE_SIZE]

    print(f"Sampling {SAMPLE_SIZE} images across {len(MODELS)} models")
    print(f"Images: {[img.name for img in images[:5]]}... (showing first 5)")
    print()

    all_results = []

    # Each model gets its own thread
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(sample_model, model, images): model
            for model in MODELS
        }
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                results = fut.result()
                all_results.extend(results)
                print(f"\n[DONE] {model} — {len(results)}/{SAMPLE_SIZE} succeeded")
            except Exception:
                print(f"\n[ERROR] {model} crashed", file=sys.stderr)
                traceback.print_exc()

    # Write results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "model", "prompt_tokens", "completion_tokens", "total_tokens"
        ])
        writer.writeheader()
        for row in sorted(all_results, key=lambda r: (r["model"], r["image"])):
            writer.writerow(row)

    print(f"\nWrote {len(all_results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
