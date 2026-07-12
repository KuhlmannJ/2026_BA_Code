# Codebase: Extracting GHG Emissions Data from ESG Reports

Codebase for the Bachelor Thesis _Beyond Text-Based RAG: Evaluating Visual RAG and Long Context for Automated GHG Emission Extraction from Sustainability Reports_ at the University of Münster (Chair for Information Systems and Business Process Management). The project extracts greenhouse gas emissions figures (Scope 1, Scope 2 market-/location-based, Scope 3) from corporate ESG reports in PDF form and compares several extraction approaches against a manually curated gold standard.

The underlying question is how much context an extraction model actually needs: passing a whole report to a long-context frontier model gives the model everything but also a lot of noise, whereas retrieving only the few relevant pages first gives it a clean but potentially incomplete view.

## Extraction approaches

All three approaches share the same extraction prompt ([baselines/baseline_frontier_model/Baseline-Prompt.txt](baselines/baseline_frontier_model/Baseline-Prompt.txt)) and the same output JSON schema, so their results are directly comparable.

| Approach | Idea |
| --- | --- |
| **Baseline** | Long context. The full report is handed to a frontier model (Claude) in one call, no retrieval step. |
| **Pipeline A** | RAG. A ColPali-style vision retriever scores every page against a retrieval query; the top pages (plus their neighbours) are cut into a mini-PDF and sent to Claude for extraction. |
| **Pipeline B** | Open-weight VLM. Same retrieval front end, but extraction runs locally on Qwen3-VL on the PALMA II HPC cluster instead of a hosted frontier model. Includes GEPA prompt optimization. |

## Repository structure

| Folder | Purpose |
| --- | --- |
| [src/pipelines/](src/pipelines/) | The extraction pipelines (`pipelineA/`, `pipelineB/`), numbered by execution order. |
| [src/GEPA/](src/GEPA/) | GEPA prompt optimization for the Pipeline B extraction prompt. |
| [src/playground/](src/playground/) | Scratch notebooks and model experiments. Not part of any pipeline. |
| [baselines/](baselines/) | The shared extraction prompt and the raw baseline (long-context) extraction results. |
| [evaluations/](evaluations/) | Gold standard, plus one folder per evaluation (see below). |
| [sh/](sh/) | SLURM batch scripts that launch the pipeline steps on PALMA II. |
| [localdata/](localdata/) | Local PDF reports and retrieval outputs (not tracked). |

## Execution order

### Pipeline A (RAG → Claude)

1. [A-01-embed.py](src/pipelines/pipelineA/A-01-embed.py) — renders each report's pages to images and embeds them with an NVIDIA ColEmbed retriever (`-3B` / `-4B` / `-8B` selects the model), writing one embedding tensor per report.
2. [A-02-retrieval.py](src/pipelines/pipelineA/A-02-retrieval.py) — embeds the retrieval query, scores it against the stored page embeddings, and writes the top-scoring pages (with ±1 neighbours) out as a mini-PDF per report. `--query` overrides the default query.
3. [A-03-toClaude.py](src/pipelines/pipelineA/A-03-toClaude.py) — submits the mini-PDFs to the Anthropic **batch** API with the shared extraction prompt and stores the returned batch ID.
4. [A-04-fromClaude.py](src/pipelines/pipelineA/A-04-fromClaude.py) — polls that batch, then writes one extraction JSON per report plus token/cost totals.

[A-01-embed-ColPali.py](src/pipelines/pipelineA/A-01-embed-ColPali.py) embeds with the original `vidore/colpali-v1.3` model. It is early exploration only — the ColEmbed script above superseded it, and it is not used in the pipeline.

### Pipeline B (RAG → Qwen3-VL)

Steps 1 and 2 are identical to Pipeline A: [B-01-embed-ColEmbed3Bv2.py](src/pipelines/pipelineB/B-01-embed-ColEmbed3Bv2.py) and [B-02-retrieval.py](src/pipelines/pipelineB/B-02-retrieval.py) are empty stubs that point back at `A-01`/`A-02`. Only the extraction step differs:

3. [B-03-HPC.py](src/pipelines/pipelineB/B-03-HPC.py) — runs the extraction locally on a Qwen3-VL checkpoint on PALMA. `-m` picks the variant (`think`, `moe`, `instr`, `instrFP8`, `instr8B`), `-mt` caps thinking tokens, and `--prompt` swaps in a different prompt file (this is how GEPA-optimized prompts are evaluated).
4. [B-03-UniGPT.py](src/pipelines/pipelineB/B-03-UniGPT.py) — the same extraction step against the university's OpenAI-compatible UniGPT endpoint (Gemma / Qwen) instead of a local checkpoint.

### Baseline

The raw long-context extractions live in [baselines/baseline_frontier_model/raw/](baselines/baseline_frontier_model/raw/) and are consumed directly by the evaluation notebooks. They were manually generated via the [Claude web interface](https://claude.ai/).

### GEPA prompt optimization

[src/GEPA/oa_main.py](src/GEPA/oa_main.py) drives a `gepa.optimize_anything` loop that mutates the extraction prompt, starting from the seed prompt in [Query0_Extraction.txt](src/GEPA/Query0_Extraction.txt). Each candidate prompt is scored by [oa_evaluate.py](src/GEPA/oa_evaluate.py), which runs the Qwen3-VL extraction via [B_03_HPC_fn.py](src/GEPA/B_03_HPC_fn.py) (the `B-03-HPC` logic exposed as importable functions) over a small training set of reports and matches the result against the gold standard using [oa_mapping.py](src/GEPA/oa_mapping.py). Every candidate prompt and its score is kept under `evaluations/GEPA_Prompt_Optimization/GEPA_runs/`.

## Evaluation

The gold standard is [evaluations/gs_slim.json](evaluations/gs_slim.json) — a slimmed-down version of `gold_standard.csv`, restricted to the reports actually available locally and produced by [gs_slimming.py](evaluations/gs_slimming.py).

Each evaluation follows the same two-notebook pattern:

- **`01-Prep-*.ipynb`** — flattens the nested extraction JSONs into a table (via the folder's `flattening.py`), merges them onto the gold standard on report/scope/year, and writes the merged frame to a `*_ynorm.json` intermediate.
- **`02-Eval-*.ipynb`** — reads that intermediate and computes hit rates per variant and per category, using two matching criteria: `check_hit` (lenient) and `check_hit_exactly` (strict).

| Evaluation | What it compares |
| --- | --- |
| [baseline/](evaluations/baseline/) | Long-context baseline against the gold standard. |
| [PipelineA/](evaluations/PipelineA/) | Pipeline A (retrieval + Claude) against the gold standard. |
| [PipelineB/](evaluations/PipelineB/) | Pipeline B against the gold standard, across the different Qwen3-VL variants. Uses `01-ReferenceDFs.ipynb` / `02-Evaluation.ipynb` rather than the `01-Prep` / `02-Eval` names. |
| [GEPA_Prompt_Optimization/](evaluations/GEPA_Prompt_Optimization/) | The GEPA-optimized prompts against the seed prompt, per model and run. |
| [Baseline-PipelineA/](evaluations/Baseline-PipelineA/) | Baseline vs. Pipeline A head-to-head — the long-context vs. RAG comparison. |
| [Baseline-PipelineA-PipelineB/](evaluations/Baseline-PipelineA-PipelineB/) | All three approaches together against the gold standard, with Pipeline B represented by its GEPA-optimized run. |

Two further folders evaluate the retrieval stage itself rather than the extractions: [A-01/](evaluations/A-01/) compares embedding throughput across the 3B/4B/8B retrievers and batch sizes, and [A-02/](evaluations/A-02/) checks whether the retrieved pages actually contain the gold-standard figures (`retrieval_misses.csv`).

## Setup

The work is split across two environments, each with its own pinned requirements file.

| Environment | File | Used for |
| --- | --- | --- |
| **Local** | [requirements-local.txt](requirements-local.txt) | The evaluation notebooks, the gold standard scripts, and the Claude batch steps `A-03` / `A-04`. CPU only — no `torch`. |
| **HPC** | [requirements-HPC.txt](requirements-HPC.txt) | Everything that needs a GPU on PALMA II: embedding (`A-01`), retrieval (`A-02`), Qwen3-VL extraction (`B-03-HPC`), and GEPA optimization. |

```bash
# local (data analysis / notebooks)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt

# on PALMA (GPU steps)
ml palma/2024a GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.2
python -m venv ~/venvs/<name> && source ~/venvs/<name>/bin/activate
pip install -r requirements-HPC.txt
```

- **Python**: the HPC environment is pinned to 3.12 (the `flash_attn` wheel in `requirements-HPC.txt` is a prebuilt `cp312` / CUDA 13 / torch 2.12 binary and will not install on another combination, thanks to [the prebuild wheels by mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels)). The local environment was built on 3.13.
- **Not covered**: `colpali-engine` is in neither file. It is only needed for [A-01-embed-ColPali.py](src/pipelines/pipelineA/A-01-embed-ColPali.py), which is early exploration rather than part of the pipeline; the file's header comment notes the extra install.
- **Credentials**: API keys are read from a local `.env` (not tracked) via `python-dotenv`.
- **Running on the cluster**: the GPU steps are submitted as SLURM jobs — see [sh/](sh/). Those scripts, and the scratch paths inside the Python files, are hardcoded to the author's cluster account and will need adjusting before they run anywhere else.
