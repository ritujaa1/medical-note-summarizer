"""
Medical Note Summarizer — Streamlit Community Cloud edition (free tier).

Runs a Qwen2.5-0.5B sibling model fine-tuned on the same dataset as the
headline Llama 2 7B (which is too large for free hosting — see
https://huggingface.co/ritujaa1/llama2-med-summarizer-gguf). The 0.5B GGUF
is ~400 MB and fits the free tier's ~1 GB RAM.
"""

import os
import re

import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Medical Note Summarizer", page_icon="🩺", layout="wide")

FT_REPO = os.getenv("MODEL_REPO", "ritujaa1/qwen05-med-summarizer-gguf")
FT_FILE = os.getenv("MODEL_FILE", "qwen05-med-summarizer-Q4_K_M.gguf")

REQUIRED_SECTIONS = ["DIAGNOSIS", "KEY FINDINGS", "MEDICATIONS", "ACTION ITEMS", "FOLLOW-UP"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

PROMPT_TEMPLATE = (
    "You are a clinical documentation assistant. Summarize the following "
    "clinical note into a structured summary with exactly these sections: "
    "DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP. "
    "Be concise and factual; do not add information not present in the note.\n\n"
    "CLINICAL NOTE:\n{note}\n\nSTRUCTURED SUMMARY:"
)


@st.cache_resource(show_spinner=False)
def load_model():
    from llama_cpp import Llama

    path = hf_hub_download(repo_id=FT_REPO, filename=FT_FILE)
    return Llama(model_path=path, n_ctx=1024, n_threads=2, n_threads_batch=2, verbose=False)


def truncate(text: str) -> str:
    for marker in ("</s>", "<|endoftext|>", "CLINICAL NOTE:", "You are a clinical"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"(FOLLOW-UP:\s*[A-Za-z0-9\- ]*).*", r"\1", text, flags=re.S)
    text = re.sub(r"[^\x20-\x7E\n]+", "", text)
    return text.strip()


def summarize(note: str) -> str:
    llm = load_model()
    out = llm(
        PROMPT_TEMPLATE.format(note=note.strip()),
        max_tokens=300,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["CLINICAL NOTE:", "\nYou are"],
    )
    return truncate(out["choices"][0]["text"])


def structure_compliance(prediction: str) -> float:
    present = sum(1 for s in REQUIRED_SECTIONS if s in prediction.upper())
    return present / len(REQUIRED_SECTIONS)


def number_fidelity(source_note: str, prediction: str) -> float:
    source_nums = set(NUM_RE.findall(source_note))
    pred_nums = NUM_RE.findall(prediction)
    if not pred_nums:
        return 1.0
    return sum(1 for n in pred_nums if n in source_nums) / len(pred_nums)


st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 340px); }
.block-container { padding-top: 3.5rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }
.med-hero {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 60%, #38bdf8 100%);
    border-radius: 18px; padding: 2.2rem 2.4rem 2rem; color: #fff;
    margin-top: .8rem; margin-bottom: 1.6rem; box-shadow: 0 8px 24px rgba(2,132,199,.25);
}
.med-hero h1 { color:#fff; font-size:2rem; font-weight:700; margin:0 0 .35rem; padding:0; letter-spacing:-.02em; }
.med-hero p { color:#e0f2fe; font-size:1.02rem; margin:0; }
.med-badges { margin-top:1.2rem; display:flex; flex-wrap:wrap; gap:.7rem; }
.med-badge {
    display:inline-block; background:rgba(255,255,255,.16);
    border:1px solid rgba(255,255,255,.35); border-radius:999px;
    padding:.35rem 1rem; font-size:.78rem; font-weight:600; color:#fff;
}
.med-card {
    background:#fff; border:1px solid #bae6fd; border-radius:14px;
    padding:1.1rem 1.3rem; margin-bottom:.8rem; box-shadow:0 2px 8px rgba(14,165,233,.07);
}
.med-card-title { color:#0369a1; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
.med-card-body { color:#0f172a; font-size:.95rem; line-height:1.55; }
.med-model-head {
    display:flex; align-items:center; gap:.5rem; font-size:1.12rem; font-weight:700;
    color:#0f172a; padding-bottom:.5rem; border-bottom:3px solid #0ea5e9; margin-bottom:.9rem;
}
.med-chip { font-size:.7rem; font-weight:700; border-radius:999px; padding:.18rem .6rem; text-transform:uppercase; letter-spacing:.05em; background:#e0f2fe; color:#0369a1; }
.med-metric { display:inline-block; border-radius:12px; padding:.55rem 1rem; margin:.6rem .6rem 0 0; border:1px solid #bae6fd; background:#f0f9ff; }
.med-metric .v { font-size:1.25rem; font-weight:700; color:#0284c7; }
.med-metric .l { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.05em; }
.med-disclaimer {
    background:#fffbeb; border:1px solid #fde68a; border-radius:12px;
    padding:.8rem 1.1rem; font-size:.85rem; color:#92400e; margin-top:1.4rem;
}
.stTextArea textarea { border:1.5px solid #bae6fd !important; border-radius:12px !important; font-size:.92rem !important; }
.stTextArea textarea:focus { border-color:#0ea5e9 !important; box-shadow:0 0 0 3px rgba(14,165,233,.18) !important; }
.stButton button {
    background:linear-gradient(135deg,#0284c7,#0ea5e9) !important; color:#fff !important;
    border:none !important; border-radius:10px !important; padding:.55rem 1.6rem !important;
    font-weight:600 !important; box-shadow:0 4px 12px rgba(2,132,199,.3) !important;
}
.stButton button:hover { filter:brightness(1.07); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="med-hero">
  <h1>🩺 Medical Note Summarizer</h1>
  <p>Turn free-text clinical notes into clean, structured summaries — LoRA-fine-tuned
     language models trained on synthetic clinical data.</p>
  <div class="med-badges">
    <span class="med-badge">✓ Structured 5-section output</span>
    <span class="med-badge">✓ Hallucination checks</span>
    <span class="med-badge">✓ Free cloud demo (0.5B model)</span>
  </div>
</div>
""", unsafe_allow_html=True)

SAMPLE_NOTE = """Patient is a 58-year-old male presenting for follow-up of blood sugar control.
HPI: Reports increased thirst and fatigue. Also notes blurred vision.
PMH: Hyperlipidemia, obesity.
Social history: Former smoker, quit 5 years ago. No alcohol.
Vitals: BP 138/86, HR 82, afebrile.
Objective/Results: HbA1c 8.2%. Fasting glucose 162 mg/dL.
Current medications include metformin 1000 mg BID.
Assessment: Findings consistent with type 2 diabetes mellitus.
Plan: add empagliflozin 10 mg daily; repeat HbA1c in 3 months; reinforce dietary counseling. Follow up in 3 months."""

SECTION_ICONS = {"DIAGNOSIS": "🔬", "KEY FINDINGS": "📋", "MEDICATIONS": "💊",
                 "ACTION ITEMS": "✅", "FOLLOW-UP": "📅"}
SECTION_RE = re.compile(
    r"^(DIAGNOSIS|KEY FINDINGS|MEDICATIONS|ACTION ITEMS|FOLLOW-UP):\s*(.*)$", re.M)


def render_summary(text: str) -> None:
    matches = SECTION_RE.findall(text)
    if len(matches) >= 3:
        for name, body in matches:
            icon = SECTION_ICONS.get(name, "•")
            st.markdown(
                f'<div class="med-card"><div class="med-card-title">{icon} {name}</div>'
                f'<div class="med-card-body">{body or "—"}</div></div>',
                unsafe_allow_html=True)
    else:
        st.code(text, language=None)


st.markdown("#### 📄 Clinical note")
note = st.text_area("Paste a clinical note", value=SAMPLE_NOTE, height=260,
                    label_visibility="collapsed")

run = st.button("✨ Summarize", type="primary")

if run and note.strip():
    st.markdown('<div class="med-model-head">Structured summary '
                '<span class="med-chip">Fine-tuned Qwen2.5-0.5B</span></div>',
                unsafe_allow_html=True)
    with st.spinner("Generating (typically 15-60 s on free hardware)..."):
        try:
            ft = summarize(note)
            render_summary(ft)
            s = structure_compliance(ft)
            nf = number_fidelity(note, ft)
            st.markdown(
                f'<div class="med-metric"><div class="v">{s:.0%}</div><div class="l">Structure compliance</div></div>'
                f'<div class="med-metric"><div class="v">{nf:.0%}</div><div class="l">Number fidelity</div></div>',
                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Model error: {e}")

st.markdown("""
<div class="med-disclaimer">
  ⚠️ <strong>Educational demo only.</strong> Trained on fully synthetic data — do not use
  with real patient information or for clinical decisions. This free demo runs a 0.5B
  sibling model; the headline
  <a href="https://huggingface.co/ritujaa1/llama2-med-summarizer-gguf">Llama 2 7B fine-tune</a>
  is too large for free hosting.
  <em>Structure compliance</em> = fraction of the 5 required sections present.
  <em>Number fidelity</em> = fraction of numbers in the summary that appear in the source note.
</div>
""", unsafe_allow_html=True)
