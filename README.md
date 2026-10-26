# 🩺 Medical Note Summarizer — Fine-tuned Llama 2 7B

Fine-tunes **Llama 2 7B** with **LoRA, locally on Apple Silicon (MLX)**, to convert free-text clinical notes into structured summaries with fixed sections: **DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP**. Includes synthetic dataset generation, a full evaluation framework (ROUGE, BLEU, structure compliance, hallucination checks), a Streamlit comparison app, and Hugging Face publishing.

> ⚠️ Educational/portfolio project. Trained on fully synthetic data. Not for real patient data or clinical decisions.

**Live demo:** [medical-note-summarizer-ritujaa1.streamlit.app](https://medical-note-summarizer-ritujaa1.streamlit.app) (Qwen2.5-0.5B sibling model, same data and pipeline) · **Model:** [ritujaa1/llama2-med-summarizer-gguf](https://huggingface.co/ritujaa1/llama2-med-summarizer-gguf)

![Architecture](docs/architecture.png)

## Architecture

```
generate_dataset.py ──> train/val/test JSONL (prompt/completion format)
        │
        ▼
train.py ──> Replicate managed fine-tuning (LoRA on Llama 2 7B, cloud GPUs)
        │
        ▼
inference.py ──> replicate.run() against base + fine-tuned versions
        │
        ▼
evaluate.py ──> ROUGE / BLEU / structure compliance / number fidelity
        │                                   │
        ▼                                   ▼
streamlit_app.py (side-by-side demo)   push_to_hf.py (HF dataset + model card)
```

## Quickstart (free, fully local — Apple Silicon)

Trains a LoRA adapter on the 4-bit quantized base model with [mlx-lm](https://github.com/ml-explore/mlx-lm). No cloud account, no cost. Needs ~16 GB RAM and ~20 GB disk.

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate + validate data (data/mlx/ holds the mlx-lm copies)
python data/generate_dataset.py --n 600 --out data/
python src/data_prep.py --file data/train.jsonl

# 3. Download + quantize the base model (~13 GB download, 3.5 GB on disk)
python -m mlx_lm convert --hf-path NousResearch/Llama-2-7b-hf \
    --mlx-path models/llama2-7b-4bit -q

# 4. Train the LoRA adapter (~30-60 min on an M-series Mac)
python src/train_local.py

# 5. Evaluate base vs fine-tuned on the held-out test set
python src/evaluate.py --mode live --n 25

# 6. Demo app
streamlit run app/streamlit_app.py
```

`INFERENCE_BACKEND=mlx` is the default; set it to `replicate` in `.env` to use the cloud path below instead.

## Quickstart (Replicate cloud training, ~$3-7)

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in your tokens

# 2. Generate + validate data
python data/generate_dataset.py --n 600 --out data/
python src/data_prep.py --file data/train.jsonl

# 3. Host train.jsonl at a public URL
#    Easiest: create a GitHub gist, upload train.jsonl, copy the "Raw" link
#    into TRAIN_DATA_URL in .env

# 4. Create an empty destination model at https://replicate.com/create
#    (e.g. llama2-med-summarizer) and set DESTINATION_MODEL in .env
#    Verify the current trainable version hash at
#    https://replicate.com/meta/llama-2-7b/train and set BASE_MODEL_VERSION.

# 5. Train (~15-40 min on Replicate's GPUs; a few dollars of credit)
python src/train.py
#    When it finishes, copy the printed version into FINETUNED_MODEL in .env

# 6. Evaluate base vs fine-tuned on the held-out test set
python src/evaluate.py --mode live --n 25

# 7. Demo app
streamlit run app/streamlit_app.py

# 8. Publish dataset + model card to Hugging Face
python deploy/push_to_hf.py --username <your-hf-username>
```

## Evaluation framework

| Metric | What it measures | Why it matters |
|---|---|---|
| ROUGE-1/2/L | n-gram overlap with reference summary | Standard summarization quality |
| BLEU | Precision-oriented overlap | Complements ROUGE (recall-oriented) |
| Structure compliance | % of the 5 required sections present | The actual product requirement |
| Number fidelity | % of numbers in output that exist in the source note | Hallucination check for invented dosages/vitals — critical in clinical text |

Results are written to `results/eval_report.json` and printed as a comparison table. The base model typically fails structure compliance badly (it chats instead of following the format) while the fine-tuned model learns the exact output schema — this gap is your headline resume metric.

## Project structure

```
medical-note-summarizer/
├── data/generate_dataset.py    # synthetic SOAP notes -> JSONL
├── src/
│   ├── config.py               # env-driven configuration
│   ├── data_prep.py            # format validation + stats
│   ├── train.py                # Replicate training kickoff + monitoring
│   ├── inference.py            # base/fine-tuned inference
│   └── evaluate.py             # metrics + comparison report
├── app/streamlit_app.py        # side-by-side demo UI
├── deploy/push_to_hf.py        # HF dataset + model card publishing
├── docs/                       # resume bullets + interview guide
└── results/                    # predictions + eval reports
```

## Cost estimate

- Training: Llama 2 7B LoRA on Replicate ≈ **$2–6** for this dataset size
- Evaluation (50 inference calls): ≈ **$0.50–1**
- Everything else runs locally (M-series MacBook is plenty)

## Key design decisions (see docs/INTERVIEW_GUIDE.md for the full story)

- **Managed LoRA over local training** — spend engineering time on data quality and evaluation, not GPU babysitting
- **Prompt/completion format with a strict output schema** — makes quality objectively measurable (structure compliance)
- **Custom hallucination metric** — generic ROUGE won't catch an invented dosage; number fidelity will
- **Synthetic data** — sidesteps PHI/HIPAA issues entirely while keeping the task realistic
