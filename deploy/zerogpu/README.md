---
title: Medical Note Summarizer
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.12"
sdk_version: "6.30.0"
app_file: app.py
pinned: false
license: llama2
models:
- ritujaa1/llama2-med-summarizer-gguf
- TheBloke/Llama-2-7B-GGUF
---

# 🩺 Medical Note Summarizer

Llama 2 7B fine-tuned with LoRA to convert free-text clinical notes into structured
summaries (DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP).

Runs the fused Q4_K_M GGUF (dequantized to fp16 by transformers) on ZeroGPU —
a few seconds per summary.

> ⚠️ Educational demo trained on fully synthetic data. Not for real patient data
> or clinical decisions.

Fine-tuned vs base evaluation on 25 held-out synthetic notes: ROUGE-1 0.944 vs 0.710,
BLEU 0.920 vs 0.351, structure compliance 1.000 vs 0.976.

Source: https://github.com/ritujaa1/medical-note-summarizer
