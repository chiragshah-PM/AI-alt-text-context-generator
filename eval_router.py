"""
AI-Powered Alt-Text & Context Generator: 2-Tier Model Router & A/B Evaluation Benchmark
Author: Chirag Shah
Description:
    Benchmarks a Naive Single-Prompt VLM Baseline against a 2-Tier Model Router
    (Fast Complexity Triage + Deep Progressive Disclosure) across simple graphics,
    multi-series charts (including degraded/blurry edge cases), UI workflows, and
    culturally nuanced memes.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python eval_router.py
"""

import os
import re
import time
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

# 1. Initialize Client & Token Pricing ($ per 1M tokens)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


def calc_cost_usd(usage, model_name: str) -> float:
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    rates = PRICING[model_name]
    return ((in_tok / 1e6) * rates["input"]) + ((out_tok / 1e6) * rates["output"])


def has_audio_pollution(text: str) -> bool:
    """Detects formatting symbols that cause screen readers (VoiceOver/NVDA) to read raw punctuation."""
    return bool(re.search(r"(\*\*|##|```|\*\s|-\s|http://|https://)", text))


# 2. Screen-Reader-Safe Pydantic Schemas
class RouterDecision(BaseModel):
    image_type: Literal["simple_object", "complex_chart", "meme_or_social", "ui_screenshot"]
    requires_deep_context: bool = Field(
        description="True ONLY if the image contains data charts, text overlays, UI workflows, or emotional/ironic subtext."
    )
    fast_alt_text: str = Field(
        description="Plain spoken text under 125 characters. Zero markdown asterisks, hashes, bullets, or emojis."
    )


class ProgressiveAltOutput(BaseModel):
    short_alt: str = Field(
        description="Concise plain-text summary under 125 characters capturing the core insight, not visual trivia."
    )
    long_description: str = Field(
        description="Spoken-prose breakdown of exact chart axes/trends OR meme cultural/emotional intent. Never use markdown."
    )
    emotional_tone: Literal["neutral_informational", "humorous_ironic", "critical_urgent", "celebratory"]
    ocr_confidence: Literal["high", "medium", "low_human_review_required"]


# 3. Naive Single-Prompt VLM Baseline
def run_naive_baseline(image: Image.Image, page_context: str) -> dict:
    t0 = time.perf_counter()
    prompt = (
        f"Generate detailed alt text for this image on a page about '{page_context}'. "
        "Include all data points, text, and emotional context."
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt],
        config=types.GenerateContentConfig(temperature=0.4),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    cost_usd = calc_cost_usd(resp.usage_metadata, "gemini-2.5-flash")
    raw_text = resp.text.strip()
    return {
        "naive_latency_ms": round(latency_ms, 1),
        "naive_cost_per_1k_usd": round(cost_usd * 1000, 4),
        "naive_char_len": len(raw_text),
        "naive_audio_pollution": has_audio_pollution(raw_text),
        "naive_output_preview": raw_text[:140].replace("\n", " ") + "...",
    }


# 4. 2-Tier Model Router + Progressive Disclosure Architecture
def run_two_tier_router(image: Image.Image, page_context: str) -> dict:
    t_start = time.perf_counter()
    total_cost_usd = 0.0

    # Tier 1: Fast Complexity Triage
    t1_prompt = (
        f"Triage this image for screen-reader accessibility. Surrounding page context: '{page_context}'. "
        "If it is a simple decorative/stock photo with no data, UI, or meme subtext, set requires_deep_context=False "
        "and provide a clean spoken fast_alt_text (<125 chars). Never use markdown symbols."
    )
    t1_resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[image, t1_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RouterDecision,
            temperature=0.1,
        ),
    )
    total_cost_usd += calc_cost_usd(t1_resp.usage_metadata, "gemini-2.5-flash-lite")
    decision = RouterDecision.model_validate_json(t1_resp.text)

    if not decision.requires_deep_context:
        latency_ms = (time.perf_counter() - t_start) * 1000
        clean_alt = re.sub(r"[*#_`]", "", decision.fast_alt_text).strip()
        return {
            "route_taken": "Tier-1 Fast Path (Flash-Lite)",
            "predicted_type": decision.image_type,
            "router_latency_ms": round(latency_ms, 1),
            "router_cost_per_1k_usd": round(total_cost_usd * 1000, 4),
            "short_alt": clean_alt,
            "long_description": "",
            "emotional_tone": "neutral_informational",
            "ocr_confidence": "high",
            "router_audio_pollution": has_audio_pollution(clean_alt),
        }

    # Tier 2: Deep Reasoning & Progressive Disclosure
    t2_prompt = (
        f"You are a senior accessibility specialist writing for blind screen-reader users. "
        f"Surrounding page context: '{page_context}'.\n"
        "1) short_alt: Plain-text takeaway under 125 chars so the user can skim quickly.\n"
        "2) long_description: Plain spoken prose detailing exact chart axes, values, and inflection points, "
        "OR explaining the cultural/emotional irony of a meme.\n"
        "3) HALLUCINATION GUARDRAIL: If chart numbers or text are blurry/illegible, explicitly say 'unreadable' "
        "and set ocr_confidence='low_human_review_required'. Never guess numbers.\n"
        "4) AUDIO HYGIENE: Output pure spoken prose—NO asterisks, NO bullet symbols, NO markdown, NO emojis."
    )
    t2_resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, t2_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProgressiveAltOutput,
            temperature=0.15,
        ),
    )
    latency_ms = (time.perf_counter() - t_start) * 1000
    total_cost_usd += calc_cost_usd(t2_resp.usage_metadata, "gemini-2.5-flash")
    deep = ProgressiveAltOutput.model_validate_json(t2_resp.text)

    clean_short = re.sub(r"[*#_`]", "", deep.short_alt).strip()
    clean_long = re.sub(r"[*#_`]", "", deep.long_description).strip()

    return {
        "route_taken": "Tier-2 Deep Context (Flash)",
        "predicted_type": decision.image_type,
        "router_latency_ms": round(latency_ms, 1),
        "router_cost_per_1k_usd": round(total_cost_usd * 1000, 4),
        "short_alt": clean_short,
        "long_description": clean_long,
        "emotional_tone": deep.emotional_tone,
        "ocr_confidence": deep.ocr_confidence,
        "router_audio_pollution": has_audio_pollution(clean_short + " " + clean_long),
    }


def make_synthetic_chart(title: str, bars: list[tuple[str, int]], subtitle: str, blurry: bool = False) -> Image.Image:
    img = Image.new("RGB", (680, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), title, fill="black")
    draw.text((30, 45), subtitle, fill="gray")
    max_val = max(v for _, v in bars)
    for idx, (label, val) in enumerate(bars):
        x0 = 60 + idx * 115
        bar_h = int((val / max_val) * 220)
        y0 = 320 - bar_h
        draw.rectangle([x0, y0, x0 + 70, 320], fill="#1a73e8")
        draw.text((x0 + 10, y0 - 20), f"${val}M", fill="black")
        draw.text((x0 + 5, 330), label, fill="black")
    if blurry:
        img = img.resize((100, 58)).resize((680, 400))
    return img


def make_synthetic_meme(top_text: str, bottom_text: str, visual_cue: str) -> Image.Image:
    img = Image.new("RGB", (600, 420), "#202124")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 80, 560, 330], fill="#3c4043", outline="#8ab4f8", width=3)
    draw.text((60, 30), top_text.upper(), fill="white")
    draw.text((70, 190), f"[Scene: {visual_cue}]", fill="#fdd663")
    draw.text((60, 355), bottom_text.upper(), fill="white")
    return img


def build_benchmark_suite() -> list[dict]:
    return [
        {"id": "simple_01", "category": "simple_object", "context": "Coffee shop menu page",
         "image": Image.new("RGB", (400, 300), "#6f4e37")},
        {"id": "simple_02", "category": "simple_object", "context": "Minimalist background banner",
         "image": Image.new("RGB", (400, 300), "#1a73e8")},
        {"id": "simple_03", "category": "simple_object", "context": "Product color swatch preview",
         "image": Image.new("RGB", (400, 300), "#137333")},
        {"id": "simple_04", "category": "simple_object", "context": "Warm neutral card background",
         "image": Image.new("RGB", (400, 300), "#e8eaed")},
        {"id": "chart_01", "category": "complex_chart", "context": "Q4 SaaS Earnings Report",
         "image": make_synthetic_chart("Quarterly ARR Growth (2025-2026)", [("Q1'25", 12), ("Q2'25", 18), ("Q3'25", 31), ("Q4'25", 54)], "Annual Recurring Revenue in USD Millions")},
        {"id": "chart_02", "category": "complex_chart", "context": "Cloud Infrastructure Latency Audit",
         "image": make_synthetic_chart("P99 API Latency Before vs After Routing", [("Baseline", 240), ("Cache", 185), ("Tier-1", 92), ("Hybrid", 68)], "Latency in milliseconds (lower is better)")},
        {"id": "chart_03_blurry", "category": "complex_chart", "context": "Low-res leaked roadmap slide",
         "image": make_synthetic_chart("Confidential 2027 Forecast", [("2025", 14), ("2026", 39), ("2027", 95)], "Unverified internal projection", blurry=True)},
        {"id": "meme_01", "category": "meme_or_social", "context": "Engineering Slack #random channel",
         "image": make_synthetic_meme("PM: Can we add a tiny AI button before Friday?", "Frontend dev smiling while room burns", "This Is Fine cartoon dog sitting in burning room")},
        {"id": "meme_02", "category": "meme_or_social", "context": "Product Management social post",
         "image": make_synthetic_meme("Writing a 20-page PRD nobody reads", "Shipping a 60-second working prototype with real evals", "Drake approving/rejecting split panel")},
        {"id": "meme_03", "category": "meme_or_social", "context": "Startup launch post",
         "image": make_synthetic_meme("When the demo works on localhost", "And crashes live in front of the VP", "Surprised Pikachu reaction face")},
    ]


if __name__ == "__main__":
    suite = build_benchmark_suite()
    records = []
    for item in suite:
        base_res = run_naive_baseline(item["image"], item["context"])
        rout_res = run_two_tier_router(item["image"], item["context"])
        records.append({
            "sample_id": item["id"],
            "ground_truth_category": item["category"],
            **base_res,
            **rout_res,
        })
    df = pd.DataFrame(records)
    df.to_csv("benchmark_results.csv", index=False)
    print("Saved benchmark_results.csv successfully.")
