# Codebase Bachelor Thesis "Beyond Text-Based RAG: Evaluating Visual RAG and Long Context for Automated GHG Emission Extraction from Sustainability Reports"

Codebase for the bachelor's thesis **"Beyond Text-Based RAG: Evaluating Visual RAG and Long Context for Automated GHG Emission Extraction from Sustainability Reports"** (University of Münster, Department of Information Systems, 2026).

The thesis compares four end-to-end approaches for extracting Scope 1–3 greenhouse gas emissions from corporate sustainability reports and evaluates them against a gold-standard dataset. This repository holds the code for all four, the prompt-optimization experiment, the SLURM scripts used on the PALMA II cluster, and the evaluation notebooks.

<p align="center">
  <img src="Readme_Approaches.svg" alt="Overview of the four approaches" width="750">
</p>

---

## Why four approaches

Modern frontier models hold context windows large enough to swallow an entire sustainability report, which raises the question whether a retrieval step is still needed at all. To test that, the **Baseline** hands the full report to a frontier model (Long Context). **Pipeline A** keeps the same extractor but feeds it only the ~9 pages a visual retriever selected, isolating the effect of the page budget. **Pipeline B** replaces the proprietary extractor with an open-weight VLM that reads rendered page images, testing whether a locally hosted model can compete. **Pipeline B_G** adds a GEPA-optimized extraction prompt on top of Pipeline B.

| | Baseline | Pipeline A | Pipeline B | Pipeline B_G |
| --- | --- | --- | --- | --- |
| Retrieval model | — | ColEmbed-8B | ColEmbed-8B | ColEmbed-8B |
| Page budget | all pages | ≤ 9 | ≤ 9 | ≤ 9 |
| Page representation | PDF | PDF | page images, 150 DPI | page images, 150 DPI |
| Extraction model | Claude Opus 4.7 | `claude-opus-4-7` | Qwen3-VL-32B-Thinking | Qwen3-VL-32B-Thinking |
| Extraction prompt | P₀ | P₀ | P₀ | P_GEPA |
| Execution environment | Claude web interface (manual) | Anthropic Batch API | PALMA II (H200) | PALMA II (H200) |

Retrieval uses **Nemotron ColEmbed-VL 8B V2** with `TOP_K = 3`, expanded by the ±1 neighbour pages, which yields retrieval sets of 5–9 pages (mean 7.6) from reports averaging ~85 pages.

All four approaches use the same extraction prompt P₀ (`baselines/baseline_frontier_model/Baseline-Prompt.txt`, except B_G) and emit the same JSON schema, so their outputs are flattened and scored on one grid:

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

## Results

Scored over the 54 reports of `gs_slim`. The evaluation grid has 2,208 report–Scope–year cells, of which 489 carry a reported value and 1,719 are legitimately empty.

| Approach | Value recall (any) | Value recall (exact) | Reports fully correct (of 54) |
| --- | :---: | :---: | :---: |
| Baseline | 89.57 % | 86.09 % | 39 (72.2 %) |
| Pipeline A | 92.23 % | 90.59 % | 41 (75.9 %) |
| Pipeline B | 90.80 % | 86.50 % | 38 (70.4 %) |
| Pipeline B_G | **93.05 %** | 89.98 % | **44 (81.5 %)** |

Retrieval over the 72 gold-standard (report, page) pairs: **Recall@3 = 91.67 %** on the top-3 pages alone, **98.61 %** after the ±1 neighbour expansion, with no report missed entirely.

Full tables, the per-Scope breakdown, the error-type analysis and the cost/latency figures are in the thesis (Chapter 6); the notebooks that produce them are listed under [Evaluation](#evaluation).

---

## Repository structure

```
.
├── baselines/
│   └── baseline_frontier_model/
│       ├── Baseline-Prompt.txt        # P0 — extraction prompt used by all approaches
│       └── raw/                       # one Baseline JSON per report (54), hand-collected
├── localdata/                         # report PDFs, retrieval sets, retrieval logs
├── sh/                                # SLURM batch scripts (PALMA II)
├── src/
│   ├── pipelines/
│   │   ├── pipelineA/                 # A-01 … A-04
│   │   └── pipelineB/                 # B-03 (embedding/retrieval reuse Pipeline A)
│   ├── GEPA/                          # prompt optimization via gepa.optimize_anything
│   └── colpali-original.py            # reference script
├── evaluations/                       # gold standard, flattening, notebooks, results
├── requirements-HPC.txt               # pip freeze from the HPC venv
└── requirements-local.txt             # pip freeze from the local venv
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt      # on the cluster: requirements-HPC.txt
```

Both requirement files are `pip freeze` dumps of the environments actually used. The HPC one pins a prebuilt `flash_attn` wheel and CUDA 13 builds and is **not** meant to be installed locally.

Secrets are read from a gitignored `.env` (`python-dotenv`):

```
ANTHROPIC_API_KEY=...     # Pipeline A (A-03/A-04)
OPENAI_API_KEY=...        # UniGPT endpoint (GEPA reflection LM)
OPENAI_API_BASE=...
```

> **Cluster paths are hardcoded.** The pipeline scripts read and write under `SCRATCH_ROOT = /scratch/tmp/jkuhlma1`, and the SLURM scripts assume the repository sits at `$HOME/2026_BA_Code`. Both must be adjusted before running anywhere else. Results committed to this repository were copied back from `SCRATCH_ROOT` into `localdata/`, `evaluations/` and `baselines/`.

---

## Reproducing the results

The steps below run in order. Every script takes `--help`; the flags are documented there rather than duplicated here.

**0 — Data.** `src/playground/fundamentals/extractReports.py` downloads the report PDFs listed in `usefulURLs.csv` and logs failures. `evaluations/gs_slimming.py` then builds `gs_slim.json` from `gold_standard.csv`, applying the corrections to known gold-standard errors described in the thesis (§4.1.2).

**1 — Baseline.** Not reproducible from this repository: each report was uploaded by hand to the Claude web interface and the returned JSON was copied into `baselines/baseline_frontier_model/raw/`. The committed JSONs are the record of that run (see [Caveats](#caveats)).

**2 — Embedding and retrieval** (shared by Pipeline A and B):

```bash
sbatch sh/A-01-embed.sh -8B        # page embeddings → one .pt per report + KPI log
sbatch sh/A-02-retrieval.sh -8B    # MaxSim scoring → mini-PDF per report + retrieval log
```

`A-01` renders pages with PyMuPDF at 150 DPI and runs ColEmbed in bf16 with FlashAttention-2 at batch size 8. `A-02` embeds the retrieval query Q₀ with the same model, scores every page via MaxSim, and writes the top-3 pages plus their ±1 neighbours as a mini-PDF. The model flag is required — there is no default.

**3 — Pipeline A** (extraction via the Anthropic Batch API):

```bash
python src/pipelines/pipelineA/A-03-toClaude.py    # submit the batch
python src/pipelines/pipelineA/A-04-fromClaude.py  # poll, write one JSON + usage log per report
```

Three API conditions are available via `-c`: `bare`, `thinking`, and `thinking_system` (the default, and the condition reported in the thesis, since it comes closest to the web interface the Baseline ran in). Each writes to its own output folder so batch IDs and skip logic do not mix.

**4 — Pipeline B** (extraction with a local VLM):

```bash
sbatch sh/B-03-HPC.sh -m t         # dense Qwen3-VL-32B-Thinking (reference extractor)
```

`-m` also accepts the MoE (`m`) and non-thinking Instruct (`i`) variants. The script renders the retrieval-set pages, runs the extraction prompt with a 16,384-token budget (retried once at double the budget), strips the reasoning trace at `</think>` along with code fences, and writes one JSON per report plus a results CSV with runtime and pages per report.

**5 — Prompt optimization** (produces Pipeline B_G):

```bash
sbatch sh/GEPA-01_H200.sh          # src/GEPA/oa_main.py
sbatch sh/B-03-HPC.sh -m t -p src/GEPA/oa_result.txt
```

`oa_main.py` seeds `gepa.optimize_anything` with P₀ and optimizes against the 60 % training split, using GPT-OSS 120B as the external reflection model. Per-iteration outputs land in `evaluations/GEPA_Prompt_Optimization/GEPA_runs/<run>/<iteration>/`. The split is seeded (`random.seed(42)`) and stratified by whether the unoptimized model already extracted a report correctly, so the optimizer sees both easy and hard reports.

**6 — Evaluation.** Each comparison folder under `evaluations/` follows the same two-notebook pattern: `01-Prep-*.ipynb` flattens the extractions, joins them onto `gs_slim.json` and normalizes fiscal years into a `*_ynorm.json`; `02-Eval-*.ipynb` reads that file and computes the metrics. Notebook 01 must run first.

---

## Evaluation

| Folder | Content |
| --- | --- |
| `baseline/`, `PipelineA/`, `PipelineB/` | per-approach preparation and evaluation |
| `Baseline-PipelineA/`, `Baseline-PipelineA-PipelineB/` | cross-approach comparisons (headline results) |
| `GEPA_Prompt_Optimization/` | per-run preparation, run outputs, summaries |
| `A-01/` | embedding-model KPI comparison (3B/4B/8B), batch-size comparison |
| `A-02/` | retrieval evaluation against the gold-standard pages |

Shared helpers: `gs_slimming.py` (builds `gs_slim` incl. gold-standard fixes), `flattening.py`, `gs_pageCount.py` / `gs_slim_pageCount.py`, `gepa_split_valuebearing.py` (training/held-out split analysis), `cost_analysis.py` (Pipeline A cost from the A-04 usage logs, Pipeline B latency from the B-03 result CSVs).

---

## Data

| Path | Content |
| --- | --- |
| `localdata/esg_reports_all/` | the 114 downloadable reports of the gold-standard dataset |
| `localdata/esg_reports/` | the 54 reports with extractable emission values (`gs_slim`) |
| `localdata/esg_reports_gepaTrainSet/` | retrieval sets flagged as the GEPA training split |
| `localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/` | the 54 retrieval sets produced by A-02 |
| `localdata/A-02-retrieval_log.csv`, `failed_urls.csv` | logs |

Of the 139 reports referenced by the gold standard, 25 are no longer downloadable and 60 of the remaining 114 contain no extractable emission values, leaving the 54 reports used throughout (thesis §4.1.1). The report PDFs remain the property of the reporting companies; re-download them with `extractReports.py` rather than expecting them in the git history.

---

## Caveats

Points that bound how the numbers above should be read; all are discussed in the thesis (§6, §7.6).

- **The Baseline is not a reproducible experiment.** It ran through the consumer web interface, which applies a provider-side system prompt, may invoke tools, and can change at any time. The same report submitted twice is not guaranteed to be the same experiment.
- **Single runs.** Baseline, Pipeline A and Pipeline B_G were each run once. Only Pipeline B has a variance estimate (two runs per model).
- **The neighbour expansion inflates the retrieval hit rate.** It exists because the gold standard records the page number printed in the report, not the physical PDF page.
- **Precision is not reported.** The gold standard is knowingly incomplete for non-total Scope 3 values, so an extracted value absent from the dataset is not necessarily an error.
- **Units are scored, not converted.** Values are compared exactly as reported.

---

## Dataset citation

The gold-standard emission values are derived from:

> Beck, J., Steinberg, A., Dimmelmeier, A., Domenech Burin, L., Kormanyos, E., Fehr, M., & Schierholz, M. (2025). Addressing data gaps in sustainability reporting: A benchmark dataset for greenhouse gas emission extraction. *Scientific Data*, 12(1), 1497. https://doi.org/10.1038/s41597-025-05664-8