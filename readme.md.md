# AI-Powered Alt-Text & Context Generator

> **Cutting VLM Inference Latency by 47.7% and API Unit Costs by 52.4% via 2-Tier Model Routing & Progressive Disclosure**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Accessibility: WCAG 2.2 AA](https://img.shields.io/badge/Accessibility-WCAG%202.2%20AA-green.svg)](#architecture)

## Overview
Modern web communication relies heavily on information-dense visuals—multi-series charts, UI screenshots, and culturally nuanced memes—that break traditional automated alt-text systems. Running flagship Vision-Language Models (VLMs) on 100% of web images is prohibitively slow and expensive at browser/workspace scale, while dumping 300-word descriptions into `alt` tags creates severe audio fatigue for screen-reader users.

This repository implements a **2-Tier VLM Routing Pipeline** paired with a **Progressive Disclosure HTML Schema** (`short_alt` + `aria-describedby` inside native `<details>` elements).

## Repository Contents
- `eval_router.py` — End-to-end Python evaluation benchmark comparing a Naive Single-Prompt VLM baseline against our 2-Tier Router (`gemini-2.5-flash-lite` triage + `gemini-2.5-flash` deep reasoning).
- `benchmark_results.csv` — Quantitative A/B evaluation log measuring latency (ms), cost per 1,000 images (USD), OCR confidence fallback triggers, and audio-pollution rates.
- `test_accessibility.html` — Standalone screen-reader evaluation harness validated with macOS VoiceOver and Windows NVDA.

## Key Benchmark Highlights (10-Image Multimodal Suite)
| Pipeline Configuration | Median Latency (ms) | Mean Cost / 1,000 Images | Markdown Audio Pollution | Hallucination Guardrail |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Single-Prompt VLM** | 2,448.1 ms | $1.8700 | 80.0% (8/10) | None (guesses blurry figures) |
| **2-Tier Router + JSON Schema** | **1,279.9 ms (-47.7%)** | **$0.8900 (-52.4%)** | **0.0% (0/10)** | **Deterministic (`low_human_review_required`)** |

## Quickstart
```bash
pip install google-genai pydantic pillow requests pandas
export GEMINI_API_KEY="your-api-key"
python eval_router.py
```