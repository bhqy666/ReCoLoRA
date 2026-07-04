# HiLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning

Code and raw results for the paper *HiLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning* (arXiv link to be added).

HiLoRA is a parameter-efficient framework for continual fine-tuning of LLMs. It initializes adapters from a randomized SVD of the pretrained weight, selects per-layer effective ranks with an elbow criterion, and trains the principal subspace before opening residual capacity. Before each new task, the current effective weight is re-decomposed into a frozen residual, a slowly updated principal component, and a fresh adapter (recursive consolidation), so every task starts from the model that has already absorbed the previous ones.

## Layout

- `src/hilora/` — method implementation: `svd.py` (randomized SVD + elbow rank selection), `recursive.py` (recursive consolidation), `inject.py` (adapter injection), `vanilla.py`/`lora.py` (static variants), `olora.py` (O-LoRA baseline).
- `train_continual_llm.py` — main driver: sequential GLUE fine-tuning of a causal LM with HiLoRA or PEFT baselines (LoRA, PiSSA, AdaLoRA, DoRA, O-LoRA).
- `train_taskbank_llm.py` — HiLoRA-TaskBank (one frozen branch per task, oracle routing).
- `scripts/` — experiment launchers and aggregation:
  - `run_recursive_hilora_llm_suite.sh` — HiLoRA on the four backbones (main table).
  - `run_peft_rank_sweep_llm.sh` — baseline rank sweep (r = 64/128/256); seeds via `SEEDS="42 43 44"`.
  - `run_olora_remaining_suite.sh`, `run_extra_robustness_experiments.sh` — O-LoRA and robustness runs.
  - `legacy_unused_experiments/` — launchers for the static single-adapter studies reported in the appendix (Qwen3 ablations, Llama rank-freezing, Mistral/InternLM baselines).
  - `summarize_llm_results.py`, `aggregate_continual_results.py` — build summary tables from `outputs/`.
- `outputs/` — raw per-run result JSONs (`continual_results.json`: config, per-stage evaluation matrix, forgetting summary) for every experiment in the paper.
- `figures/` — figure sources and PDFs used in the paper; `main.tex` / `references.bib` — paper source.

## Setup

```bash
pip install -r requirements.txt
```

Backbones (Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-v0.3, InternLM2.5-7B-Chat) are loaded through Hugging Face Transformers; GLUE data through `datasets`. All experiments were run on a single consumer GPU (RTX 2080 Ti); each task trains for 200 optimizer steps at max length 256.

## Reproducing the main results

Six-task continual GLUE sequence (SST-2 → MRPC → QNLI → RTE → QQP → MNLI):

```bash
# HiLoRA (recursive consolidation) on all four backbones
bash scripts/run_recursive_hilora_llm_suite.sh

# Baseline rank sweep (LoRA/PiSSA/AdaLoRA/DoRA), seed 42
bash scripts/run_peft_rank_sweep_llm.sh

# Summaries
python scripts/summarize_llm_results.py --root outputs/continual_llm --model_tag qwen3_8b
```

Metrics (final average score, average forgetting) are computed from the task-by-stage evaluation matrix stored in each `continual_results.json`.
