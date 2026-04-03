#!/usr/bin/env python3
"""
Estimate benchmark costs per model using actual token usage and live OpenRouter pricing.

Reads token_usage.csv for observed token counts, fetches current prices from
the OpenRouter API, and projects costs for the full 400-image benchmark.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import requests

USAGE_CSV = Path("ocr-results/token_usage_sample.csv")
BENCHMARK_SIZE = 400  # total images in full benchmark


def fetch_openrouter_pricing() -> dict:
    """Fetch model pricing from OpenRouter API. Returns {model_id: {prompt, completion}}."""
    resp = requests.get("https://openrouter.ai/api/v1/models")
    resp.raise_for_status()
    models = resp.json().get("data", [])
    pricing = {}
    for m in models:
        mid = m.get("id", "")
        p = m.get("pricing", {})
        prompt_price = p.get("prompt")
        completion_price = p.get("completion")
        if prompt_price is not None and completion_price is not None:
            pricing[mid] = {
                "prompt": float(prompt_price),       # cost per token
                "completion": float(completion_price),  # cost per token
            }
    return pricing


def load_usage() -> dict:
    """Load token usage CSV. Returns {model: {images, prompt_tokens, completion_tokens}}."""
    usage = defaultdict(lambda: {"images": 0, "prompt_tokens": 0, "completion_tokens": 0})
    with open(USAGE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            pt = int(row["prompt_tokens"]) if row["prompt_tokens"] else 0
            ct = int(row["completion_tokens"]) if row["completion_tokens"] else 0
            usage[model]["images"] += 1
            usage[model]["prompt_tokens"] += pt
            usage[model]["completion_tokens"] += ct
    return dict(usage)


def main():
    print("Fetching live pricing from OpenRouter...")
    pricing = fetch_openrouter_pricing()

    print(f"Loading token usage from {USAGE_CSV}...")
    usage = load_usage()

    print()
    print(f"{'Model':<45} {'N':>4} {'Avg In':>8} {'Avg Out':>8} {'In $/M':>8} {'Out $/M':>8} {'Sample $':>9} {'Full 400 $':>10}")
    print("-" * 130)

    rows = []
    for model, u in sorted(usage.items()):
        n = u["images"]
        if n == 0:
            continue
        avg_in = u["prompt_tokens"] / n
        avg_out = u["completion_tokens"] / n

        p = pricing.get(model)
        if p:
            in_per_m = p["prompt"] * 1_000_000
            out_per_m = p["completion"] * 1_000_000
            sample_cost = u["prompt_tokens"] * p["prompt"] + u["completion_tokens"] * p["completion"]
            full_cost = (avg_in * p["prompt"] + avg_out * p["completion"]) * BENCHMARK_SIZE
        else:
            in_per_m = out_per_m = sample_cost = full_cost = None

        rows.append((model, n, avg_in, avg_out, in_per_m, out_per_m, sample_cost, full_cost))

    rows.sort(key=lambda r: r[7] if r[7] is not None else 999)

    for model, n, avg_in, avg_out, in_per_m, out_per_m, sample_cost, full_cost in rows:
        if in_per_m is not None:
            print(f"{model:<45} {n:>4} {avg_in:>8.0f} {avg_out:>8.0f} "
                  f"${in_per_m:>7.3f} ${out_per_m:>7.3f} "
                  f"${sample_cost:>8.4f} ${full_cost:>9.4f}")
        else:
            print(f"{model:<45} {n:>4} {avg_in:>8.0f} {avg_out:>8.0f} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>9} {'N/A':>10}")

    print()
    total_full = sum(r[7] for r in rows if r[7] is not None)
    print(f"Total estimated cost for all models (400 images each): ${total_full:.2f}")


if __name__ == "__main__":
    main()
