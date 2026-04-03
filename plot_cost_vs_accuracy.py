#!/usr/bin/env python3
"""
Scatter plot: estimated cost (400 images) vs. overall accuracy (1 - CER alnum).
Combines token_usage_sample.csv + OpenRouter live pricing + ocr_eval_model_accuracy.csv.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

USAGE_CSV = Path("ocr-results/token_usage_sample.csv")
ACCURACY_CSV = Path("ocr_eval_model_accuracy.csv")
BENCHMARK_SIZE = 1000
OUTPUT_PNG = Path("cost_vs_accuracy.png")
OUTPUT_FACET_PNG = Path("cost_vs_accuracy_by_type.png")

# Map sanitized dir names back to OpenRouter model IDs
def sanitize_for_path(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")

# Models to skip (local/non-API, broken, or no data)
SKIP = {
    "Tesseract", "EasyOcr", "qwen3vl_finetuned", "qwen3vl_unsloth4b",
    "qwen-qwen-vl-maxmistralai-mistral-medium-3.1",
    "nvidia-nemotron-nano-12b-v2-vl-free",
}

# Provider color map
PROVIDER_COLORS = {
    "google": "#4285F4",
    "openai": "#10A37F",
    "anthropic": "#D97706",
    "qwen": "#7C3AED",
    "meta-llama": "#1877F2",
    "mistralai": "#FF6B35",
    "amazon": "#FF9900",
    "microsoft": "#00BCF2",
    "nvidia": "#76B900",
    "moonshotai": "#E91E63",
    "openrouter": "#888888",
    "x-ai": "#000000",
}


def fetch_pricing() -> dict:
    resp = requests.get("https://openrouter.ai/api/v1/models")
    resp.raise_for_status()
    pricing = {}
    for m in resp.json().get("data", []):
        mid = m.get("id", "")
        p = m.get("pricing", {})
        pp, cp = p.get("prompt"), p.get("completion")
        if pp is not None and cp is not None:
            pricing[mid] = {"prompt": float(pp), "completion": float(cp)}
    return pricing


def load_usage() -> dict:
    usage = defaultdict(lambda: {"n": 0, "prompt_tokens": 0, "completion_tokens": 0})
    with open(USAGE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            pt = int(row["prompt_tokens"]) if row["prompt_tokens"] else 0
            ct = int(row["completion_tokens"]) if row["completion_tokens"] else 0
            usage[model]["n"] += 1
            usage[model]["prompt_tokens"] += pt
            usage[model]["completion_tokens"] += ct
    return dict(usage)


def load_accuracy() -> dict:
    """Returns {sanitized_model: {"Overall": float, "Book Page": float, ...}}"""
    acc = {}
    with open(ACCURACY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            vals = {}
            for col in ["Overall", "Book Page", "Handwritten", "Mixed", "Other Typed/Printed"]:
                v = row.get(col, "")
                if v:
                    try:
                        vals[col] = float(v)
                    except ValueError:
                        pass
            if vals:
                acc[model] = vals
    return acc


def main():
    print("Fetching pricing...")
    pricing = fetch_pricing()
    usage = load_usage()
    accuracy = load_accuracy()

    # Build model_id -> sanitized name mapping
    id_to_sanitized = {}
    for model_id in usage:
        id_to_sanitized[model_id] = sanitize_for_path(model_id)

    # Match usage (by model_id) to accuracy (by sanitized name)
    points = []
    for model_id, u in usage.items():
        sanitized = id_to_sanitized[model_id]
        if sanitized in SKIP or model_id in SKIP:
            continue

        acc_dict = accuracy.get(sanitized)
        if acc_dict is None:
            continue
        overall = acc_dict.get("Overall")
        if overall is None or overall < 0:
            continue

        p = pricing.get(model_id)
        if p is None:
            continue

        n = u["n"]
        avg_in = u["prompt_tokens"] / n
        avg_out = u["completion_tokens"] / n
        cost = (avg_in * p["prompt"] + avg_out * p["completion"]) * BENCHMARK_SIZE

        provider = model_id.split("/")[0] if "/" in model_id else "other"
        short_name = model_id.split("/")[1] if "/" in model_id else model_id

        point = {
            "model_id": model_id,
            "short_name": short_name,
            "provider": provider,
            "cost": cost,
            "Overall": overall * 100,
        }
        for doc_type in ["Book Page", "Handwritten", "Mixed", "Other Typed/Printed"]:
            v = acc_dict.get(doc_type)
            point[doc_type] = v * 100 if v is not None and v > 0 else None
        points.append(point)

    def plot_scatter(ax, points, acc_key, title):
        for pt in points:
            val = pt.get(acc_key)
            if val is None:
                continue
            color = PROVIDER_COLORS.get(pt["provider"], "#888888")
            ax.scatter(pt["cost"], val, c=color, s=60, alpha=0.8,
                       edgecolors="white", linewidth=0.5, zorder=3)
            ax.annotate(
                pt["short_name"], (pt["cost"], val),
                fontsize=5.5, alpha=0.8, xytext=(5, 3), textcoords="offset points",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0, top=105)

    # Overall plot
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_scatter(ax, points, "Overall", "InkBench: OCR Accuracy vs. Cost")
    ax.set_xlabel("Estimated Cost per 1,000 Images ($)", fontsize=12)
    ax.set_ylabel("Accuracy (%, 1 − CER alphanumeric)", fontsize=12)

    seen = set()
    handles = []
    for pt in sorted(points, key=lambda p: p["provider"]):
        if pt["provider"] not in seen:
            seen.add(pt["provider"])
            color = PROVIDER_COLORS.get(pt["provider"], "#888888")
            handles.append(plt.scatter([], [], c=color, s=60, label=pt["provider"]))
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Provider")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    plt.close()

    # Faceted by document type
    doc_types = ["Book Page", "Handwritten", "Mixed", "Other Typed/Printed"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, doc_type in zip(axes.flat, doc_types):
        plot_scatter(ax, points, doc_type, doc_type)
        ax.set_xlabel("Cost per 1,000 Images ($)", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)

    # Shared legend
    seen = set()
    handles = []
    for pt in sorted(points, key=lambda p: p["provider"]):
        if pt["provider"] not in seen:
            seen.add(pt["provider"])
            color = PROVIDER_COLORS.get(pt["provider"], "#888888")
            handles.append(plt.scatter([], [], c=color, s=60, label=pt["provider"]))
    fig.legend(handles=handles, loc="lower center", ncol=len(seen), fontsize=8, title="Provider")

    fig.suptitle("InkBench: OCR Accuracy vs. Cost by Document Type", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(OUTPUT_FACET_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_FACET_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
