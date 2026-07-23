# 2026_BA_Code

Codebase for the bachelor's thesis on AI-based extraction of greenhouse gas (GHG) emissions data from corporate sustainability reports.

The repository contains three extraction setups, the prompt-optimization experiment around them, the shell/SLURM scripts used to run everything on the PALMA II HPC cluster, and the notebooks used to evaluate the results against a gold standard.

Four end-to-end approaches are compared for extracting Scope 1–3 GHG emissions: a Long Context **Baseline** that hands the frontier model the full report, two **RAG pipelines** (**Pipeline A**: frontier model, **Pipeline B**: open-weight VLM) that first retrieve the ~9 most relevant pages via visual late interaction retrieval (ColEmbed), and a **GEPA-optimized** variant of Pipeline B's extraction prompt. The goal is to see whether retrieval is still needed once a model's context window is large enough to hold an entire report (see thesis, Chapter 1).

---

## Contents

- [Extraction setups](#extraction-setups)
- [Repository structure](#repository-structure)
- [Pipelines](#pipelines)
- [Prompt optimization (GEPA)](#prompt-optimization-gepa)
- [Evaluation](#evaluation)
- [Data](#data)
- [Setup](#setup)
- [Running on PALMA II](#running-on-palma-ii)
- [Conventions](#conventions)

---

## Extraction setups

| Setup | Retrieval | Extraction | Where it runs |
|---|---|---|---|
| **Baseline** | none (full report) | frontier model (Claude Opus), `Baseline-Prompt.txt` | — |
| **Pipeline A** | ColEmbed (`nvidia/nemotron-colembed-vl-*`) | Claude via the Anthropic Batch API | retrieval on HPC, extraction via API |
| **Pipeline B** | ColEmbed (same retrievals as A) | Qwen3-VL, run locally on the cluster | HPC (H200) |

Pipeline B additionally has a variant (`B-03-UniGPT.py`) that sends the same retrieved pages to models served via an OpenAI-compatible endpoint (UniGPT).

All setups use the same extraction prompt (`baselines/baseline_frontier_model/Baseline-Prompt.txt`) and produce the same JSON schema, so their outputs can be flattened and compared against the same gold standard.

### Output schema

```json
{
  "report_name": "<filename without .pdf>",
  "report_title": "<full report title>",
  "emissions": {
    "scope_1":                { "<year>": [{ "value": 0, "unit": "", "label": "" }] },
    "scope_2_market_based":   { "<year>": [ ... ] },
    "scope_2_location_based": { "<year>": [ ... ] },
    "scope_3":                { "<year>": [ ... ] }
  }
}
```

---

## Repository structure

```
.
├── baselines/
│   └── baseline_frontier_model/
│       ├── Baseline-Prompt.txt        # extraction prompt used by all setups
│       └── raw/                       # one baseline JSON per report (54)
├── localdata/                         # PDFs, retrieved page PDFs, retrieval log
├── sh/                                # SLURM batch scripts (PALMA II)
├── src/
│   ├── pipelines/
│   │   ├── pipelineA/                 # A-01 … A-04
│   │   └── pipelineB/                 # B-03 (embedding/retrieval reuse Pipeline A)
│   ├── GEPA/                          # prompt optimization via gepa.optimize_anything
│   ├── colpali-original.py            # reference script
│   └── playground/                    # exploratory scripts/notebooks, not part of the pipelines
├── evaluations/                       # gold standard, flattening, notebooks, results
├── requirements-HPC.txt               # pip freeze from the HPC venv
└── requirements-local.txt             # pip freeze from the local venv
```

---

## Pipelines

### Pipeline A — `src/pipelines/pipelineA/`

| Step | Script | Purpose |
|---|---|---|
| A-01 | `A-01-embed.py` | Renders report pages and stores per-report ColEmbed page embeddings as `.pt`; writes a KPI log (runtime, s/page, peak VRAM/RAM, file size) |
| A-02 | `A-02-retrieval.py` | Scores pages against the retrieval query, selects Top-*k*, writes a mini-PDF of the selected pages plus a Top-10 JSON and a retrieval log |
| A-03 | `A-03-toClaude.py` | Builds and submits the Anthropic message batch over the mini-PDFs |
| A-04 | `A-04-fromClaude.py` | Polls the batch, writes one JSON per report and per-report token-usage logs |

Key parameters:

- **Model selection (A-01/A-02):** `-3B`, `-4B`, `-8B` → `nvidia/llama-nemotron-colembed-vl-3b-v2`, `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2`. No default; a flag is required.
- **Retrieval (A-02):** `TOP_K = 3`, expanded with ±1 neighbor pages; default query `QUERY_0` (overridable with `-q`).
- **Conditions (A-03/A-04):** `-c bare | thinking | thinking_system` (default `thinking_system`). Each condition writes to its own subfolder so batch IDs and skip-logic do not mix. `thinking_system` additionally sends `system-prompt.txt`.
- `A-03-toClaude-SysP.py` is the earlier single-condition variant of `A-03-toClaude.py`.
- **Data-set flags:** `-t` (test path), `-a` (all reports), `-gt` (GEPA training set).

### Pipeline B — `src/pipelines/pipelineB/`

`B-01-embed.py` and `B-02-retrieval.py` are pointers to the Pipeline A scripts — B reuses A's embeddings and retrievals.

- **`B-03-HPC.py`** — loads a Qwen3-VL model, renders the retrieved mini-PDF pages at `DPI = 150`, runs the extraction prompt, strips `<think>` blocks and code fences, and writes one JSON per report plus a `***results.csv` with model, maxToken, report, duration, pages and t_inf/page.
  - `-m think | moe | instr | instrFP8 | instr8B`
  - `-mt` max (thinking) tokens, default `16384`
  - `-p` custom prompt file, `-o` custom output directory, `-t` test path, `-gt` GEPA training set
- **`B-03-UniGPT.py`** — same input, extraction through an OpenAI-compatible endpoint (`gemma-4-31B-it`, `Qwen3.5-35B-A3B`, …); model selected in-file.

---

## Prompt optimization (GEPA)

`src/GEPA/` optimizes the extraction prompt with `gepa.optimize_anything`.

| File | Purpose |
|---|---|
| `oa_main.py` | Objective, reflection LM (`openai/gpt-oss-120b` via UniGPT), seed prompt, optimizer configuration |
| `oa_evaluate.py` | Evaluator: loads the VLM once, runs extraction over the training set, scores hit rate against `evaluations/gs_slim.json`, logs each run |
| `oa_mapping.py` | Flattens extraction JSONs and maps them onto the gold standard (incl. RegEx year normalization, e.g. `FY 2021/2022`) |
| `B_03_HPC_fn.py` | `load_model` / `run_extraction` — the B-03 logic as importable functions |
| `Query0_Extraction.txt` | Seed prompt |
| `oa_result.txt` (in `pipelines/pipelineB/`) | Resulting prompt |

Runs and per-iteration outputs are stored under `evaluations/GEPA_Prompt_Optimization/GEPA_runs/<run>/<iteration>/` together with the `prompt.txt` of that iteration.

---

## Evaluation

`evaluations/` holds the gold standard and its derivatives, plus one folder per comparison. Most folders follow the same two-notebook pattern:

1. **`01-Prep-*.ipynb`** — runs `flattening.py`, merges the extractions onto the slimmed gold standard (`gs_slim.json`), normalizes years, writes a `*_ynorm.json`.
2. **`02-Eval-*.ipynb`** — reads that `*_ynorm.json` and computes the metrics/figures. Requires notebook 01 to have been run.

Folders:

| Folder | Content |
|---|---|
| `baseline/`, `PipelineA/`, `PipelineB/` | per-setup preparation, evaluation and answers |
| `Baseline-PipelineA/`, `Baseline-PipelineA-PipelineB/` | cross-setup comparisons |
| `GEPA_Prompt_Optimization/` | per-run preparation/evaluation, run outputs, summaries |
| `A-01/` | embedding-model KPI comparison (3B/4B/8B), batch-size comparison |
| `A-02/` | retrieval evaluation (hits/misses against the gold-standard pages) |

Shared helpers: `gs_slimming.py` (builds `gs_slim` from `gold_standard.csv`, incl. fixes to known gold-standard errors), `gs_pageCount.py` / `gs_slim_pageCount.py`, `cost_analysis.py` (Pipeline A cost from the A-04 usage logs; Pipeline B latency from the B-03 result CSVs).

---

## Key results

Value-bearing recall (any-matcher) over the 489 cells that carry a reported
value in `gs_slim`, full breakdown in thesis Table 5 / `evaluations/Baseline-PipelineA-PipelineB/`:

| Approach                 | Value recall (any)  | Reports fully correct (of 54)    |
| ------------------------ | :-----------------: | :------------------------------: |
| Baseline                 | 89.57%              | 39 (72.2%)                       |
| Pipeline A               | 92.23%              | 41 (75.9%)                       |
| Pipeline B (unoptimized) | 90.80%              | 38 (70.4%)                       |
| Pipeline B + GEPA        | 93.05%              | 44 (81.5%)                       |

Retrieval quality (§6.1): Recall@3 is 91.67% before and 98.61% after ±1
neighbor-page expansion, over the 72 report-page pairs in `gs_slim`.

---

## Data

`localdata/` (not tracked as a package — see paths below):

| Path | Content |
|---|---|
| `esg_reports/` | The used report-set during our work |
| `esg_reports_all/` | All 114 downloadable report |
| `esg_reports_gepaTrainSet/` | The generated retrieval set, specifically flagged as the GEPA training set |
| `A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/` | The generated retrieval set from A-02 (54) |
| `A-02-retrieval_log.csv`, `failed_urls.csv` | logs |

`src/playground/fundamentals/extractReports.py` downloads the report PDFs from `usefulURLs.csv` and logs failed downloads.

#### Dataset citation

The gold-standard emission values under `localdata/` and `evaluations/` are
derived from:

> Beck, J., Steinberg, A., Dimmelmeier, A., Domenech Burin, L., Kormanyos, E.,
> Fehr, M., & Schierholz, M. (2025). Addressing data gaps in sustainability
> reporting: A benchmark dataset for greenhouse gas emission extraction.
> *Scientific Data*, 12(1), 1497. https://doi.org/10.1038/s41597-025-05664-8

The sustainability report PDFs themselves remain the property of the
reporting companies and are not redistributed beyond what the original
dataset provides.
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt   # or requirements-HPC.txt on the cluster
```

Both requirement files are `pip freeze` dumps of the environments actually used; the HPC one pins a prebuilt `flash_attn` wheel and CUDA 13 builds and is not meant to be installed locally.

Secrets are loaded from a local `.env` (`python-dotenv`, gitignored):

```
ANTHROPIC_API_KEY=...     # Pipeline A (A-03/A-04)
OPENAI_API_KEY=...        # UniGPT endpoint (B-03-UniGPT, GEPA reflection LM)
OPENAI_API_BASE=...
```

---

## Running on PALMA II

The `sh/` scripts are SLURM batch scripts. They load the modules (`palma/2024a`, `GCCcore/13.3.0`, `Python/3.12.3`, `CUDA/13.0.2`), activate the venv, set `HF_HOME`/`CUDA_HOME`/`PIP_CACHE_DIR`, and call the Python entry point.

| Script | Job | Partition / time |
|---|---|---|
| `A-01-embed.sh` | `A-01-embed.py` (arg 1 = model flag, arg 2 = mode) | `gpuh200`, 1 h |
| `A-02-retrieval.sh` | `A-02-retrieval.py` + `evaluations/A-02/A-02.py` | `gpuh200mini`, 5 min |
| `B-03-HPC.sh` | `B-03-HPC.py` (args passed through) | `gpuh200`, 8 h |
| `GEPA-01.sh` / `GEPA-01_H200.sh` | `src/GEPA/oa_main.py` | `gpua100`, 30 min / `gpuh200`, 6 d |
| `FromPDF2Extract.sh` | A-01 → A-02 → B-03 end-to-end on the test path | `gpuh200`, 10 min |
| `quick-for-nok.sh` | repeated B-03 test runs across models | `gpuh200`, 30 min |

```bash
sbatch sh/A-01-embed.sh -8B
sbatch sh/A-02-retrieval.sh -8B
sbatch sh/B-03-HPC.sh -m think
```

Some scripts also write a `pip freeze` of the job environment to `$WORK/requirements/<job>/`.

---

## Conventions

- **Cluster paths are hardcoded** to `SCRATCH_ROOT = /scratch/tmp/jkuhlma1` (data, embeddings, results, logs) and `$HOME/2026_BA_Code` in the SLURM scripts. Both must be adjusted to run elsewhere.
- **HPC vs. repo:** the pipeline scripts read/write under `SCRATCH_ROOT`; results committed here were copied into `localdata/`, `evaluations/` and `baselines/`.
- **Reports are identified by filename stem** (`<company>_<year>_report`) throughout — PDFs, JSONs and the gold standard.
- **Steps skip work that already exists** (A-03 skips reports with an existing JSON, A-01 skips existing `.pt` files).
- `banner()` and the `#### N. STEP` comments exist purely for log readability.
- Folders named `old/`, files prefixed `zzz_`/`OLD` and `src/playground/` are earlier states kept for reference; they are not part of the current runs.
