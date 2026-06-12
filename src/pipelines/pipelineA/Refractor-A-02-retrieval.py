"""
A-02-retrieval.py — ColEmbed page retrieval + mini-PDF extraction.

For each ESG-report PDF, loads pre-computed ColEmbed page embeddings,
scores them against the query, selects the top-k pages (±1 neighbours),
and writes a mini-PDF containing only those pages.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader, PdfWriter
from transformers import AutoModel


# ── Named constants ───────────────────────────────────────────────────────────

MODEL_NAME: str    = "nvidia/llama-nemotron-colembed-vl-3b-v2"
TOP_K: int         = 3          # Top-k pages to retrieve (Beck et al.)
TIME_ROUND: int    = 6          # Decimal places for timing logs
PHASE: str         = "REFRACTOR"
LOG_FORMAT: str    = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DIR: Path      = Path("logs")

# Retrieval query — placeholder for optimize_anything / GEPA optimization.
# Source: Beck et al. (Nature Dataset)
QUERY_DEFAULT: str = (
    "What are the total CO2 emissions in different years? "
    "Include Scope 1, Scope 2, and Scope 3 emissions if available."
)

# Scratch-storage layout
_SCRATCH: Path = Path("/scratch/tmp/jkuhlma1")

PDF_DIR_FULL: Path        = _SCRATCH / "data" / "esg_reports"
PDF_DIR_TEST: Path        = _SCRATCH / "data" / "test_esg_reports"
PDF_DIR_GEPA: Path        = _SCRATCH / "data" / "training" / "test_esg_reports"
EMB_DIR_DEFAULT: Path     = _SCRATCH / "data" / "embeddings" / "embeddings_colembed_3b_v2"
RETRIEVALS_DIR_DEFAULT: Path = _SCRATCH / "results" / "A-02-retrievals"
RETRIEVAL_LOG_DEFAULT: Path  = _SCRATCH / "results" / "A-02-retrieval_log.csv"

_LOG_CSV_HEADER: list[str] = [
    "report", "phase", "top_k_pages", "timestamp", "run_ts", "top_10", "top_10_scores",
]


# ── Logging factory ───────────────────────────────────────────────────────────

def get_logger(name: str, run_ts: str) -> logging.Logger:
    """Return a logger with a StreamHandler (INFO) and a FileHandler (DEBUG)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Prevent duplicate handlers on re-import

    logger.setLevel(logging.DEBUG)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(LOG_FORMAT)

    stream_h = logging.StreamHandler()
    stream_h.setLevel(logging.INFO)
    stream_h.setFormatter(fmt)

    file_h = logging.FileHandler(LOG_DIR / f"{name}_{run_ts}.log", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(stream_h)
    logger.addHandler(file_h)
    return logger


def _banner(log: logging.Logger, title: str) -> None:
    """Log a visual section separator at INFO level."""
    sep = "=" * 60
    log.info(sep)
    log.info(f"  {title}")
    log.info(sep)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _init_retrieval_log(log_path: Path) -> None:
    """Create or overwrite the CSV retrieval log with its header row."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(_LOG_CSV_HEADER)


def _append_retrieval_log(
    report_name: str,
    scores: torch.Tensor,
    log_path: Path,
    phase: str,
    run_ts: str,
    top_k: int,
) -> None:
    """Append one report's retrieval result to the CSV log file."""
    n = len(scores)
    topk_idx = scores.topk(min(top_k, n)).indices.tolist()
    top10    = scores.topk(min(10, n))

    with open(log_path, "a", newline="") as fh:
        csv.writer(fh).writerow([
            report_name,
            phase,
            topk_idx,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            run_ts,
            top10.indices.tolist(),
            top10.values.tolist(),
        ])


# ── Page selection helpers ────────────────────────────────────────────────────

def select_pages(scores: torch.Tensor, top_k: int) -> list[int]:
    """Return sorted page indices: top-k hits expanded with ±1 neighbours (Beck et al.)."""
    n = len(scores)
    top_idx = scores.topk(min(top_k, n)).indices.tolist()

    pages: set[int] = set()
    for idx in top_idx:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < n:
                pages.add(neighbor)
    return sorted(pages)


def extract_pages_as_pdf(
    pdf_path: Path, page_indices: list[int], save_path: Path
) -> None:
    """Write selected pages from pdf_path into a new mini-PDF at save_path."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for idx in page_indices:
        writer.add_page(reader.pages[idx])

    with open(save_path, "wb") as fh:
        writer.write(fh)


# ── GEPA JSON helper ──────────────────────────────────────────────────────────

def save_top10_results_json(
    report_name: str,
    scores: torch.Tensor,
    retrievals_dir: Path,
    query: str,
    run_ts: str,
) -> None:
    """Persist top-10 retrieval results as JSON for GEPA / optimize_anything."""
    n = len(scores)
    top10 = scores.topk(min(10, n))

    results: dict[str, Any] = {
        "report":       report_name,
        "query":        query,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_ts":       run_ts,
        "top_10_pages":  top10.indices.tolist(),
        "top_10_scores": top10.values.tolist(),
        "top_10_data": [
            {"page": idx.item(), "score": score.item()}
            for idx, score in zip(top10.indices, top10.values)
        ],
    }

    gepa_dir = retrievals_dir / "GEPA"
    gepa_dir.mkdir(parents=True, exist_ok=True)
    with open(gepa_dir / f"{report_name}_top10_results.json", "w") as fh:
        json.dump(results, fh, indent=2)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(model_name: str, log: logging.Logger) -> Any:
    """Load the ColEmbed retrieval model onto CUDA and return it in eval mode."""
    model = AutoModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()
    log.info(f"Loaded: {model_name}")
    log.info(f"VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return model


def embed_query(
    model: Any, query: str, log: logging.Logger, time_round: int
) -> tuple[Any, float]:
    """Embed the retrieval query; return (embeddings, runtime_seconds)."""
    t0 = time.time()
    query_embeddings = model.forward_queries([query], batch_size=1)
    runtime = round(time.time() - t0, time_round)
    log.info(f"Query embedded in {runtime}s")
    return query_embeddings, runtime


# ── Per-report processing ─────────────────────────────────────────────────────

def _process_report(
    pdf_path: Path,
    pt_map: dict[str, Path],
    model: Any,
    query_embeddings: Any,
    config: dict[str, Any],
    log: logging.Logger,
) -> dict[str, Any] | None:
    """Retrieve top pages for one report; save mini-PDF and GEPA JSON."""
    report_name = pdf_path.stem
    log.info(report_name)

    if report_name not in pt_map:
        log.warning(f"No embedding found for {report_name} — skipping")
        return None

    # Load pre-computed page embeddings
    log.debug("Loading image embeddings …")
    t1 = time.time()
    image_embeddings: list[torch.Tensor] = torch.load(
        pt_map[report_name], weights_only=False, map_location="cpu"
    )
    image_embeddings = [t.to("cuda") for t in image_embeddings]
    runtime_image_emb = round(time.time() - t1, config["time_round"])

    # Score pages and select top-k + neighbours
    log.debug("Computing similarity scores …")
    t2 = time.time()
    scores: torch.Tensor = model.get_scores(query_embeddings, image_embeddings)[0]

    _append_retrieval_log(
        report_name, scores,
        config["retrieval_log"], config["phase"], config["run_ts"], config["top_k"],
    )
    save_top10_results_json(
        report_name, scores,
        config["retrievals_dir"], config["query"], config["run_ts"],
    )

    topk = scores.topk(min(config["top_k"], len(scores)))
    log.info(f"Top-{config['top_k']} page indices : {topk.indices.tolist()}")
    log.info(f"Top-{config['top_k']} scores       : {topk.values.tolist()}")
    runtime_scoring = round(time.time() - t2, config["time_round"])

    pages = select_pages(scores, config["top_k"])
    log.info(f"Retrieved pages (0-indexed): {pages}  [{runtime_scoring}s]")

    # Extract selected pages as mini-PDF
    save_path = config["retrievals_dir"] / pdf_path.name
    extract_pages_as_pdf(pdf_path, pages, save_path)
    log.info(f"Mini-PDF saved → {save_path.name}")
    log.debug(f"runtime_image_emb: {runtime_image_emb}s")

    return {
        "report":       report_name,
        "pages":        pages,
        "top_k_pages":  topk.indices.tolist(),
        "top_k_scores": topk.values.tolist(),
    }


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run the A-02 retrieval pipeline and return a summary dict.

    Required config keys:
        run_ts          str   — Timestamp string (MMDD_HHMM)
        test            bool  — Log TEST-RUN banner if True
        query           str   — Retrieval query text
        model_name      str   — HuggingFace model identifier
        top_k           int   — Number of top pages to retrieve
        time_round      int   — Decimal places for timing logs
        phase           str   — Phase label written to CSV log
        pdf_dir         Path  — Directory of ESG-report PDFs
        emb_dir         Path  — Directory of .pt embedding files
        retrievals_dir  Path  — Output directory for mini-PDFs
        retrieval_log   Path  — Output path for retrieval CSV log
    """
    log = get_logger("A-02-retrieval", config["run_ts"])

    if config.get("test"):
        _banner(log, "THIS IS A TEST-RUN")

    # ── Step 0: environment ───────────────────────────────────────────────────
    _banner(log, "STEP 0: GLOBAL VARIABLES")
    log.debug(f".env loaded: {load_dotenv(find_dotenv())}")
    log.info(f"Query in use:\n{config['query']}")

    pdf_list: list[Path] = sorted(config["pdf_dir"].glob("*.pdf"))
    emb_list: list[Path] = sorted(config["emb_dir"].glob("*.pt"))
    log.info(f"PDFs in use       : {len(pdf_list)}")
    log.debug(f"Embeddings found  : {len(emb_list)}")

    _init_retrieval_log(config["retrieval_log"])
    config["retrievals_dir"].mkdir(parents=True, exist_ok=True)

    # ── Step 1: GPU ───────────────────────────────────────────────────────────
    _banner(log, "STEP 1: GPU / CUDA")
    props = torch.cuda.get_device_properties(0)
    log.info(f"GPU  : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM : {props.total_memory / 1e9:.1f} GB")
    log.info(f"UUID : {props.uuid}")

    # ── Step 2: model ─────────────────────────────────────────────────────────
    _banner(log, "STEP 2: Load Retrieval Model")
    model = load_model(config["model_name"], log)

    # ── Step 3: retrieval ─────────────────────────────────────────────────────
    _banner(log, "STEP 3: Begin Retrieval")
    t0 = time.time()
    query_embeddings, _ = embed_query(model, config["query"], log, config["time_round"])

    pt_map: dict[str, Path] = {p.stem: p for p in emb_list}
    results: list[dict[str, Any]] = []

    for pdf_path in pdf_list:
        result = _process_report(pdf_path, pt_map, model, query_embeddings, config, log)
        if result is not None:
            results.append(result)

    overall_time = round(time.time() - t0, config["time_round"])
    log.info(f"DONE — {len(results)}/{len(pdf_list)} reports processed in {overall_time}s")

    return {"processed": results, "total_time": overall_time}


# ── CLI wrapper ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments; no business logic."""
    parser = argparse.ArgumentParser(
        description="A-02 ColEmbed page retrieval and mini-PDF extraction"
    )
    parser.add_argument("--test",         "-t",  action="store_true",
                        help="Toggle testing paths")
    parser.add_argument("--gepaTrainSet", "-gt", action="store_true",
                        help="Toggle GEPA training-set reports")
    parser.add_argument("--query",        "-q",  type=str, default=QUERY_DEFAULT,
                        help="Custom retrieval query")
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    _run_ts: str = os.environ.get("RUN_TS", time.strftime("%m%d_%H%M"))

    if _args.gepaTrainSet:
        _pdf_dir = PDF_DIR_GEPA
    elif _args.test:
        _pdf_dir = PDF_DIR_TEST
    else:
        _pdf_dir = PDF_DIR_FULL

    _config: dict[str, Any] = {
        "run_ts":         _run_ts,
        "test":           _args.test,
        "query":          _args.query,
        "model_name":     MODEL_NAME,
        "top_k":          TOP_K,
        "time_round":     TIME_ROUND,
        "phase":          PHASE,
        "pdf_dir":        _pdf_dir,
        "emb_dir":        EMB_DIR_DEFAULT,
        "retrievals_dir": RETRIEVALS_DIR_DEFAULT,
        "retrieval_log":  RETRIEVAL_LOG_DEFAULT,
    }

    run(_config)