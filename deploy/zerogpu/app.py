"""
Medical Note Summarizer — Hugging Face Space (ZeroGPU) edition.

Runs the LoRA-fine-tuned Llama 2 7B on a free shared GPU. The fused Q4_K_M
GGUF is dequantized to fp16 by transformers at startup, so the Space serves
the exact weights that were evaluated, at GPU speed (seconds per summary).
"""

import html
import os
import re

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FT_REPO = os.getenv("MODEL_REPO", "ritujaa1/llama2-med-summarizer-gguf")
FT_FILE = os.getenv("MODEL_FILE", "med-summarizer-Q4_K_M.gguf")
BASE_REPO = os.getenv("BASE_REPO", "TheBloke/Llama-2-7B-GGUF")
BASE_FILE = os.getenv("BASE_FILE", "llama-2-7b.Q4_K_M.gguf")
DEVICE = "cuda" if os.getenv("SPACE_ID") or torch.cuda.is_available() else "cpu"

REQUIRED_SECTIONS = ["DIAGNOSIS", "KEY FINDINGS", "MEDICATIONS", "ACTION ITEMS", "FOLLOW-UP"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

PROMPT_TEMPLATE = (
    "You are a clinical documentation assistant. Summarize the following "
    "clinical note into a structured summary with exactly these sections: "
    "DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP. "
    "Be concise and factual; do not add information not present in the note.\n\n"
    "CLINICAL NOTE:\n{note}\n\nSTRUCTURED SUMMARY:"
)


# ---------------------------------------------------------------- inference

def load(repo: str, fname: str):
    tok = AutoTokenizer.from_pretrained(repo, gguf_file=fname)
    model = AutoModelForCausalLM.from_pretrained(
        repo, gguf_file=fname, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    return tok, model.to(DEVICE).eval()


# ZeroGPU: models go on cuda at import time; real GPU attaches inside @spaces.GPU
MODELS = {"finetuned": load(FT_REPO, FT_FILE), "base": load(BASE_REPO, BASE_FILE)}


def truncate(text: str) -> str:
    for marker in ("</s>", "CLINICAL NOTE:", "You are a clinical"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # FOLLOW-UP values are plain phrases; anything past other punctuation is junk
    text = re.sub(r"(FOLLOW-UP:\s*[A-Za-z0-9\- ]*).*", r"\1", text, flags=re.S)
    text = re.sub(r"[^\x20-\x7E\n]+", "", text)
    return text.strip()


@spaces.GPU(duration=60)
def generate(note: str, which: str) -> str:
    tok, model = MODELS[which]
    inputs = tok(PROMPT_TEMPLATE.format(note=note.strip()), return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=True, temperature=0.2, top_p=0.9,
            repetition_penalty=1.15, pad_token_id=tok.eos_token_id,
            stop_strings=["CLINICAL NOTE:", "\nYou are"], tokenizer=tok)
    return truncate(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


# ---------------------------------------------------------------- metrics

def structure_compliance(prediction: str) -> float:
    present = sum(1 for s in REQUIRED_SECTIONS if s in prediction.upper())
    return present / len(REQUIRED_SECTIONS)


def number_fidelity(source_note: str, prediction: str) -> float:
    source_nums = set(NUM_RE.findall(source_note))
    pred_nums = NUM_RE.findall(prediction)
    if not pred_nums:
        return 1.0
    return sum(1 for n in pred_nums if n in source_nums) / len(pred_nums)


# ---------------------------------------------------------------- rendering

SECTION_ICONS = {"DIAGNOSIS": "🔬", "KEY FINDINGS": "📋", "MEDICATIONS": "💊",
                 "ACTION ITEMS": "✅", "FOLLOW-UP": "📅"}
SECTION_RE = re.compile(
    r"^(DIAGNOSIS|KEY FINDINGS|MEDICATIONS|ACTION ITEMS|FOLLOW-UP):\s*(.*)$", re.M)


def render(note: str, output: str, title: str, chip: str, base: bool = False) -> str:
    matches = SECTION_RE.findall(output)
    if len(matches) >= 3:
        body = "".join(
            f'<div class="med-card"><div class="med-card-title">{SECTION_ICONS.get(n, "•")} {n}</div>'
            f'<div class="med-card-body">{html.escape(b) or "—"}</div></div>'
            for n, b in matches)
    else:
        body = f'<pre class="med-raw">{html.escape(output) or "(empty output)"}</pre>'
    s, nf = structure_compliance(output), number_fidelity(note, output)
    return (
        f'<div class="med-model-head{" base" if base else ""}">{title} '
        f'<span class="med-chip{" base" if base else " ft"}">{chip}</span></div>{body}'
        f'<div class="med-metric"><div class="v">{s:.0%}</div><div class="l">Structure compliance</div></div>'
        f'<div class="med-metric"><div class="v">{nf:.0%}</div><div class="l">Number fidelity</div></div>')


def summarize(note: str, compare_base: bool):
    if not note.strip():
        raise gr.Error("Paste a clinical note first.")
    ft = render(note, generate(note, "finetuned"), "Fine-tuned model", "Ours")
    if not compare_base:
        return ft, gr.update(value="", visible=False)
    base = render(note, generate(note, "base"), "Base Llama 2 7B", "Baseline", base=True)
    return ft, gr.update(value=base, visible=True)


# ---------------------------------------------------------------- UI

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
.med-hero {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 60%, #38bdf8 100%);
    border-radius: 18px; padding: 2.2rem 2.4rem 2rem; color: #fff;
    box-shadow: 0 8px 24px rgba(2,132,199,.25);
}
.med-hero h1 { color:#fff; font-size:2rem; font-weight:700; margin:0 0 .35rem; letter-spacing:-.02em; }
.med-hero p { color:#e0f2fe; font-size:1.02rem; margin:0; }
.med-badges { margin-top:1.2rem; display:flex; flex-wrap:wrap; gap:.7rem; }
.med-badge {
    background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35); border-radius:999px;
    padding:.35rem 1rem; font-size:.78rem; font-weight:600; color:#fff;
}
.med-card {
    background:#fff; border:1px solid #bae6fd; border-radius:14px;
    padding:1.1rem 1.3rem; margin-bottom:.8rem; box-shadow:0 2px 8px rgba(14,165,233,.07);
}
.med-card-title { color:#0369a1; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
.med-card-body { color:#0f172a; font-size:.95rem; line-height:1.55; }
.med-raw { white-space:pre-wrap; background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem; color:#0f172a; }
.med-model-head {
    display:flex; align-items:center; gap:.5rem; font-size:1.12rem; font-weight:700; color:#0f172a;
    padding-bottom:.5rem; border-bottom:3px solid #0ea5e9; margin-bottom:.9rem;
}
.med-model-head.base { border-bottom-color:#94a3b8; }
.med-chip { font-size:.7rem; font-weight:700; border-radius:999px; padding:.18rem .6rem; text-transform:uppercase; letter-spacing:.05em; }
.med-chip.ft { background:#e0f2fe; color:#0369a1; }
.med-chip.base { background:#f1f5f9; color:#475569; }
.med-metric { display:inline-block; border-radius:12px; padding:.55rem 1rem; margin:.6rem .6rem 0 0; border:1px solid #bae6fd; background:#f0f9ff; }
.med-metric .v { font-size:1.25rem; font-weight:700; color:#0284c7; }
.med-metric .l { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.05em; }
.med-disclaimer {
    background:#fffbeb; border:1px solid #fde68a; border-radius:12px;
    padding:.8rem 1.1rem; font-size:.85rem; color:#92400e;
}
"""

SAMPLE_NOTE = """Patient is a 58-year-old male presenting for follow-up of blood sugar control.
HPI: Reports increased thirst and fatigue. Also notes blurred vision.
PMH: Hyperlipidemia, obesity.
Social history: Former smoker, quit 5 years ago. No alcohol.
Vitals: BP 138/86, HR 82, afebrile.
Objective/Results: HbA1c 8.2%. Fasting glucose 162 mg/dL.
Current medications include metformin 1000 mg BID.
Assessment: Findings consistent with type 2 diabetes mellitus.
Plan: add empagliflozin 10 mg daily; repeat HbA1c in 3 months; reinforce dietary counseling. Follow up in 3 months."""

with gr.Blocks(title="Medical Note Summarizer") as demo:
    gr.HTML("""
    <div class="med-hero">
      <h1>🩺 Medical Note Summarizer</h1>
      <p>Turn free-text clinical notes into clean, structured summaries — powered by
         Llama&nbsp;2&nbsp;7B fine-tuned with LoRA.</p>
      <div class="med-badges">
        <span class="med-badge">✓ Structured 5-section output</span>
        <span class="med-badge">✓ Hallucination checks</span>
        <span class="med-badge">✓ Runs on a free GPU — seconds per summary</span>
      </div>
    </div>""")
    note = gr.Textbox(label="📄 Clinical note", value=SAMPLE_NOTE, lines=11)
    with gr.Row():
        run = gr.Button("✨ Summarize", variant="primary", scale=0, min_width=180)
        compare = gr.Checkbox(label="Also run base Llama 2 for comparison", value=False)
    with gr.Row(equal_height=False):
        ft_out = gr.HTML()
        base_out = gr.HTML(visible=False)
    gr.HTML("""
    <div class="med-disclaimer">
      ⚠️ <strong>Educational demo only.</strong> Trained on fully synthetic data — do not use
      with real patient information or for clinical decisions.
      <em>Structure compliance</em> = fraction of the 5 required sections present.
      <em>Number fidelity</em> = fraction of numbers in the summary that appear in the source
      note (a hallucination check for invented dosages and vitals).
    </div>""")
    run.click(summarize, [note, compare], [ft_out, base_out])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="sky"))
