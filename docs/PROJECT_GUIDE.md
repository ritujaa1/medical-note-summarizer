# 🩺 Medical Note Summarizer — Complete Project Guide

*The full story of what this project is, how every piece works, what went wrong,
what the numbers mean, and how to talk about it in interviews.*

---

## 1. What this project actually does (plain English)

Doctors write messy free-text notes. This project fine-tunes a large language
model so that when you paste in a clinical note, it returns a summary in **exactly
five fixed sections**:

```
DIAGNOSIS: Type 2 diabetes mellitus
KEY FINDINGS: Increased thirst; fatigue; HbA1c 8.2%; BP 138/86, HR 82
MEDICATIONS: Metformin 1000 mg bid
ACTION ITEMS: Add empagliflozin 10 mg daily; Repeat HbA1c in 3 months
FOLLOW-UP: 3 months
```

The key insight: an off-the-shelf model *can* summarize, but it won't reliably
follow a strict output format, and generic metrics won't tell you if it invented
a drug dosage. This project fixes the format problem with **fine-tuning** and
catches the invention problem with a **custom hallucination metric**.

Everything was done for **$0** on a MacBook (Apple M5, 16 GB RAM).

---

## 2. The pipeline, step by step

### Step 1 — Synthetic data generation (`data/generate_dataset.py`)

- Generates ~600 fake but realistic SOAP-style clinical notes with matching
  "gold" structured summaries.
- Output format: JSONL with `{"prompt": ..., "completion": ...}` pairs.
  The prompt contains the instruction + the note; the completion is the perfect
  5-section summary.
- **Why synthetic?** Real clinical notes are protected health information
  (PHI, HIPAA). Synthetic data sidesteps the entire legal/privacy problem while
  keeping the *task* realistic.
- Split: 510 train / 45 validation / 45 test. `src/data_prep.py` validates every
  row (correct keys, non-empty, reasonable lengths — mean ~250 tokens/example).

### Step 2 — Base model: download + quantize (`mlx_lm convert`)

- Base model: **Llama 2 7B** (7 billion parameters), downloaded as fp16
  safetensors (~13 GB) from an ungated mirror (NousResearch), since Meta's
  official repo requires a license click-through.
- Quantized to **4-bit** with Apple's **MLX** framework → 3.5 GB on disk.
- **Why quantize?** A 7B model in fp16 needs ~14 GB just for weights — too much
  for a 16 GB laptop that also needs memory for training. 4-bit quantization
  stores each weight in ~4 bits instead of 16, shrinking memory 4× with minor
  quality loss.

### Step 3 — LoRA fine-tuning (`src/train_local.py`)

This is the heart of the project.

- **LoRA (Low-Rank Adaptation)**: instead of updating all 6.7 billion weights,
  you freeze the base model and train tiny "adapter" matrices injected into
  16 of the transformer layers. Here that was **~10 M trainable parameters —
  0.148% of the model**. The adapter file is just 38 MB.
- **QLoRA-style**: the frozen base was the 4-bit quantized model, so the whole
  training run fit in ~5.3 GB of RAM.
- **Completion masking** (`--mask-prompt`): the loss is computed only on the
  summary tokens, not the prompt. The model learns to *write summaries*, not to
  *reproduce prompts*.
- Hyperparameters: batch size 2, learning rate 1e-4, gradient checkpointing.
- **Training curve**: validation loss 0.582 → 0.048 (iter 100) → 0.031 (iter 200).
  It converged so fast that the planned 600 iterations were stopped at ~300 —
  the last 300 would have cost an hour for negligible gain.
- Total wall time: ~45 minutes on the M5. Cost: $0 (vs ~$3–7 on a cloud service).

*One trick that was needed:* mlx-lm's prompt-masking path formats data through
the tokenizer's chat template, and base Llama 2 doesn't have one. Fix: add a
trivial pass-through template (`{{ message content, concatenated }}`) so
training format == inference format, byte for byte.

### Step 4 — Evaluation (`src/evaluate.py`)

Base model vs fine-tuned model, on 25 held-out test notes, 4 metric families:

| Metric | What it measures | Base Llama 2 | Fine-tuned |
|---|---|---|---|
| ROUGE-1 / ROUGE-L | word overlap with gold summary | 0.710 / 0.679 | **0.944 / 0.944** |
| BLEU | precision-oriented overlap | 0.351 | **0.920** |
| Structure compliance | % of the 5 required sections present | 0.976 | **1.000** |
| Number fidelity | % of numbers in output that exist in the source note | 0.996 | 0.970 |

**How to read this:**
- The ROUGE/BLEU jump (0.35 → 0.92 BLEU) is the headline: the fine-tuned model
  produces summaries nearly identical to the gold standard.
- Structure compliance hit exactly 100% — the model *always* emits the 5 sections.
- **Number fidelity is a custom metric** — a cheap hallucination detector:
  extract every number from the summary, check it appears in the source note.
  An invented dosage ("give 500 mg" when the note says 250 mg) is the most
  dangerous failure mode in clinical text, and ROUGE would barely notice it.
- Honest finding #1: the base model scored *better* than expected on structure
  (0.976) because the detailed prompt does a lot of work. The fine-tune's win is
  consistency + content quality, not just format.
- Honest finding #2: the fine-tuned model's number fidelity (0.970) is slightly
  *below* base — see "The hallucination bug" below.

### Step 5 — The demo app (`app/streamlit_app.py`)

- Streamlit web UI, medical sky-blue design: paste a note → side-by-side
  summaries from base and fine-tuned models, each with live structure-compliance
  and number-fidelity scores rendered as metric pills.
- Runs 100% locally via MLX — no API, no data leaves the machine (relevant
  selling point for anything medical).

### Step 6 — Model packaging for the world (GGUF)

MLX only runs on Apple Silicon. To let anyone use the model:

1. **Fuse**: merge the 38 MB LoRA adapter into the base weights
   (`mlx_lm fuse --dequantize`) → a standard fp16 HuggingFace-format model.
2. **Convert**: `convert_hf_to_gguf.py` (from llama.cpp) → GGUF, the format the
   ubiquitous llama.cpp runtime uses on any CPU/GPU.
3. **Quantize**: `llama-quantize` → Q4_K_M, 3.8 GB.
4. **Publish**: uploaded to Hugging Face with a model card:
   → `ritujaa1/llama2-med-summarizer-gguf`

### Step 7 — Deployment (the saga)

Goal: a free, permanent, public demo URL. Four attempts:

1. **HF Spaces (Streamlit SDK)** — rejected: HF removed the Streamlit SDK
   (only gradio/docker/static now).
2. **HF Spaces (Docker)** — blocked: HF now paywalls Docker/Gradio Spaces
   behind PRO ($9/mo).
3. **Streamlit Community Cloud** (deploys free from a GitHub repo) — the
   winner, but with a catch: it offers ~1 GB RAM and the 7B model needs ~4.5 GB.
4. **HF Spaces (ZeroGPU)** — the free route for the full 7B: free personal
   accounts older than 30 days can host 2 Gradio Spaces on ZeroGPU.
   `deploy/zerogpu/` loads the same Q4_K_M GGUF through transformers
   (dequantized to fp16) on a shared GPU, so summaries take seconds instead
   of minutes. Deploy with `python deploy/upload_zerogpu.py`.

**Solution: a sibling model.** Fine-tuned **Qwen2.5-0.5B** on the exact same
data with the exact same pipeline (~15 min). Its Q4_K_M GGUF is **379 MB** —
fits the free tier. Published as `ritujaa1/qwen05-med-summarizer-gguf`.
Scores: structure 0.88, number fidelity 1.00, ROUGE-1 0.845 — clearly labeled
in the demo as the small model, with a link to the 7B.

Final architecture:
```
GitHub repo (ritujaa1/medical-note-summarizer)
   └─ auto-deploys → Streamlit Community Cloud (free)
        └─ downloads 0.5B GGUF from Hugging Face at startup
             └─ llama-cpp-python CPU inference
```

Deployment bugs fixed along the way (all real interview material):
- Streamlit Cloud defaulted to Python 3.14, where llama-cpp-python has no
  prebuilt wheel → 30-min source build that failed. Fix: pin an
  `--extra-index-url` with prebuilt CPU wheels + redeploy on Python 3.13.
- GitHub push protection blocked a push because a local tool-config file
  containing an access token was accidentally committed (`git add -A`).
  Fix: remove file from commit, gitignore it, verify history clean.
  *Lesson: push protection is your friend; never blind-`add -A`.*

---

## 3. The hallucination bug (best story in the project)

**Symptom**: on a short test note with no vitals section, the model produced:

```
KEY FINDINGS: Sore throat; fever 101.2F; BP 100.2F; TSH 1.1F; BP 100.2F; TSH 1.1F; ...
```

An invented blood pressure, repeated forever.

**Root cause**: every synthetic training note contained a vitals line, so the
model learned "KEY FINDINGS always includes vitals." Given a note *without*
vitals, it invents them — and then falls into a repetition loop (the classic
failure mode of a small model pushed out of its training distribution).

**Fixes applied**:
1. **Repetition penalty 1.15** at inference — breaks the infinite loop.
2. **Output truncation rules** — the FOLLOW-UP field only ever contains plain
   phrases, so anything after unexpected punctuation is trimmed.
3. **Honest documentation** — the model card states the limitation.

**Real fix (future work)**: augment training data with notes that are missing
sections, so the model learns to omit rather than invent. This is a data
problem, not a model problem.

Note the meta-lesson: the custom **number-fidelity metric caught this** — it
exists precisely because ROUGE wouldn't have flagged an invented BP.

---

## 4. Resume bullets (real numbers, pick 3–4)

> - Fine-tuned Llama 2 7B with QLoRA (MLX, Apple Silicon) to convert clinical
>   notes into schema-constrained structured summaries, raising BLEU from 0.35
>   to 0.92 and output-structure compliance to 100% vs the base model — at $0
>   compute cost by training 0.15% of parameters in 4-bit on a laptop.
> - Designed a custom hallucination metric (number fidelity: % of numeric values
>   in the summary present in the source note) that surfaced a
>   vitals-invention failure mode generic metrics missed; mitigated via
>   repetition penalty and documented the data-distribution root cause.
> - Built an end-to-end evaluation framework (ROUGE, BLEU, structure compliance,
>   number fidelity) comparing base vs fine-tuned models on held-out data.
> - Shipped the full lifecycle: synthetic HIPAA-safe dataset generation → LoRA
>   training → adapter fusing → GGUF Q4_K_M quantization → two published
>   Hugging Face models → free public demo (Streamlit Cloud + llama.cpp CPU
>   inference), with a 0.5B distilled-scale sibling model engineered to fit
>   free-tier RAM limits.
> - Diagnosed and fixed production deployment issues: Python-version wheel
>   availability for native extensions, secret-scanning push blocks, and
>   chat-template incompatibilities between training and inference stacks.

---

## 5. Interview Q&A

**Q: Walk me through the project.**
A: (60-second version) "Clinical notes are free text; downstream systems want
structure. I fine-tuned Llama 2 7B with LoRA on ~500 synthetic note/summary
pairs so it emits five fixed sections. I trained on-device with Apple's MLX in
4-bit, evaluated base vs fine-tuned with ROUGE/BLEU plus two custom metrics —
structure compliance and a number-fidelity hallucination check — then fused the
adapter, converted to GGUF, published both models to Hugging Face, and deployed
a free public demo. BLEU went 0.35 → 0.92; structure compliance hit 100%."

**Q: Why LoRA instead of full fine-tuning?**
A: Full fine-tuning of 7B needs ~80+ GB of GPU memory (weights + gradients +
optimizer states in fp16/32). LoRA freezes the base and trains ~10M adapter
parameters — 0.15% of the model — which fit in 5 GB alongside 4-bit frozen
weights. For a narrow formatting task, that capacity is plenty, the risk of
catastrophic forgetting is lower, and the artifact is a portable 38 MB file.

**Q: Why is number fidelity a better safety metric than ROUGE here?**
A: ROUGE measures n-gram overlap against a reference. A summary that says
"250 mg" when the note says "500 mg" loses almost no ROUGE — one token differs —
but it's the most dangerous possible error in a clinical summary. Number
fidelity checks every number in the output against the source note, so invented
dosages, vitals, or lab values show up directly.

**Q: Your fine-tuned model scored *lower* than base on number fidelity. Why?**
A: The base model is conservative — it mostly copies text, so its numbers come
from the note. The fine-tuned model learned the training distribution too well:
every training note had vitals, so on notes without vitals it invents them.
It's an overfitting-to-format phenomenon. The metric caught it; the mitigation
is a repetition penalty plus data augmentation with section-missing notes as
future work. I'd rather present a 0.97 with an explanation than not know.

**Q: Why synthetic data? Isn't that a weakness?**
A: It's a deliberate trade-off. Real clinical data is PHI under HIPAA — using it
requires IRB-level controls that don't fit a portfolio project. Synthetic data
gives an unlimited, label-perfect, legally clean dataset. The weakness is
distribution narrowness — which I directly observed as the vitals-hallucination
bug — and that's exactly what you'd fix first with real (de-identified) data.

**Q: What is quantization and what did you trade away?**
A: Storing weights in 4 bits instead of 16 — 4× smaller memory, slightly noisier
weights. For this task the measured quality was fine (the 4-bit model hit 100%
structure compliance). The subtle part: I *trained* against the quantized base
(QLoRA-style), then fused the adapter into *dequantized* fp16 weights and
re-quantized to GGUF Q4_K_M for serving — so the adapter compensates for
quantization noise during training.

**Q: Why did you train a second, smaller model?**
A: Deployment constraint engineering. Free hosting tiers give ~1 GB RAM; the 7B
needs 4.5 GB. Instead of paying for hosting, I reused the identical pipeline on
Qwen2.5-0.5B (379 MB quantized) and shipped that as the public demo, honestly
labeled, with the 7B downloadable for anyone who wants full quality. It also
demonstrates the pipeline is model-agnostic.

**Q: What breaks if you push this toward production?**
A: (1) Synthetic→real distribution shift — needs real de-identified data and
clinician-validated references. (2) The eval set is 25 notes — needs hundreds,
plus per-section scoring. (3) Number fidelity is necessary but not sufficient —
it won't catch a wrong diagnosis stated without numbers; you'd add entailment
checks / clinician review. (4) PHI means on-prem or BAA-covered inference,
which is why the local-first design matters. (5) Regulatory: a summarizer that
influences care is potentially a medical device — human-in-the-loop framing.

**Q: Hardest bug?**
A: Two candidates. The vitals-hallucination loop (root cause: data
distribution; detected by my own metric). And an infrastructure one: training
crashed because mlx-lm's prompt masking requires a chat template and base
Llama 2 has none — I injected a pass-through template so the training-time and
inference-time token streams match exactly. Train/inference format mismatch is
one of the most common silent killers in LLM fine-tuning.

---

## 6. What you learned (honest list)

**ML concepts**
- LoRA/QLoRA mechanics: what's frozen, what trains, why it fits in laptop RAM,
  what an adapter artifact is, fusing adapters into base weights.
- Quantization formats and trade-offs (fp16 → MLX 4-bit → GGUF Q4_K_M), and
  training-vs-serving quantization interplay.
- Why completion masking matters; why train/inference prompt-format parity is
  critical; what chat templates actually do.
- Evaluation design: pairing generic metrics (ROUGE/BLEU) with task-specific
  ones (structure compliance) and safety-specific ones (number fidelity).
- Overfitting to data distribution: models don't learn "summarize," they learn
  "summarize notes shaped like the training notes" — and out-of-distribution
  inputs expose that immediately.
- Inference-time controls: temperature, top-p, repetition penalty, stop
  sequences, and post-processing as a legitimate defense layer.

**Engineering**
- The full model lifecycle: data → train → eval → fuse → convert → quantize →
  publish → deploy, and every format boundary in between.
- Free-tier constraint engineering: matching model size to hosting RAM, wheel
  availability to Python versions, prebuilt-wheel indexes vs source builds.
- Ops hygiene the hard way: GitHub push protection, secret scanning, why
  `git add -A` is dangerous, token scoping (fine-grained PAT permissions),
  rotating credentials that touched a chat/log.
- MLX (Apple's ML framework), llama.cpp/GGUF ecosystem, Hugging Face Hub
  publishing, Streamlit apps and their cloud deployment model.

**Meta**
- Cloud-managed training was the *plan*; on-device training was *cheaper and
  gave more control*. Knowing when the "serious" option is unnecessary is a
  skill.
- Honest metrics beat impressive metrics: the two "flaws" found (base model's
  decent structure score, fine-tune's fidelity dip) are the most interesting
  things to talk about.

---

## 7. Links

| What | Where |
|---|---|
| Code | github.com/ritujaa1/medical-note-summarizer |
| 7B model (GGUF) | huggingface.co/ritujaa1/llama2-med-summarizer-gguf |
| 0.5B demo model (GGUF) | huggingface.co/ritujaa1/qwen05-med-summarizer-gguf |
| Live demo | your Streamlit Cloud URL |
| Eval report | results/eval_report.json |

> ⚠️ Trained on fully synthetic data. Educational project — not for real patient
> data or clinical decisions.
