import csv
import subprocess
import pandas as pd
from pathlib import Path

from oa_mapping import map_to_goldstandard

# ── Paths
_HERE              = Path(__file__).parent
_SCRATCH           = Path("/scratch/tmp/jkuhlma1/gepa")
RUNS_DIR           = _SCRATCH / "runs"
GS_PATH            = _HERE.parent.parent / "evaluations" / "gs_slim.json"
EXTRACTION_SCRIPT  = _HERE.parent / "pipelines" / "pipelineB" / "B-03-HPC.py"
EVAL_LOG           = RUNS_DIR / "eval_log.csv"

CATEGORIES = ["value", "unit"]

_run_counter = 0
_best_score  = -1.0

def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_hit_with_detail(row, extraction_col, gs_col) -> tuple[bool, dict | None]:
    
    # Preparing gold-standard value(s) for comparison
    gs_val  = row[gs_col]
    if isinstance(gs_val,  list):
        gs_set = set(gs_val)
    elif not pd.isna(gs_val):
        gs_set = {gs_val}
    else:
        gs_set = set()
    
    # Preparing extracted value(s) for comparison
    ext_val = row[extraction_col]
    if isinstance(ext_val, list):
        ext_set = set(ext_val)
    elif not pd.isna(ext_val):
        ext_set = {ext_val}
    else:
        ext_set = set()
    
    
    # Compare value(s), as there could be more than one (e.g., Allianz) or just one (mostly)
    if gs_set == ext_set:
        return True, None
    return False, {
        "expected": gs_val,
        "got":      ext_val,
        "report":   row.get("report_name"),
        "scope":    row.get("scope"),
        "year":     row.get("year"),
    }

# Runs B-03-HPC.py
def _run_extraction(candidate: str, run_dir: Path) -> None:

    prompt_file = run_dir / "prompt.txt"
    prompt_file.write_text(candidate)

    subprocess.run(
        [
            "python", str(EXTRACTION_SCRIPT),
            "--model",       "instr8B",
            "--prompt-file", str(prompt_file),
            "--output-dir",  str(run_dir),
            "--gepaTrainSet",
        ],
        check=True,
    )


def evaluate(candidate: str) -> tuple[float, dict]:
    global _run_counter

    run_dir = RUNS_DIR / str(_run_counter)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    banner(_run_counter)
    
    print(f"\n[evaluate] Run #{_run_counter} — output dir: {run_dir}")
    _run_counter += 1

    # Run: B-03-HPC.py
    print(f"[evaluate] Running extraction (B-03-HPC.py)...")
    _run_extraction(candidate, run_dir)
    print(f"[evaluate] Extraction done.")

    # Run: oa_mapping.py (esentially "01-ReferenceDFs.ipynb" and "02-Evaluation.ipynb")
    print(f"[evaluate] Mapping to gold standard...")
    merged = map_to_goldstandard(run_dir, GS_PATH)
    print(f"[evaluate] Mapping done — {len(merged)} rows.")

    hits   = 0
    total  = 0
    misses = {cat: [] for cat in CATEGORIES}

    for _, row in merged.iterrows():
        for cat in CATEGORIES:
            total += 1
            hit, detail = check_hit_with_detail(row, f"{cat}_instr", cat)
            if hit:
                hits += 1
            elif detail:
                misses[cat].append(detail)

    global _best_score
    score        = hits / total if total > 0 else 0.0
    misses_clean = {cat: errs for cat, errs in misses.items() if errs}

    is_best = score > _best_score
    if is_best:
        _best_score = score

    write_header = not EVAL_LOG.exists()
    with EVAL_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "score", "hits", "total", "is_best"])
        if write_header:
            writer.writeheader()
        writer.writerow({"run": _run_counter - 1, "score": round(score, 6), "hits": hits, "total": total, "is_best": is_best})

    print(f"[evaluate] Score: {score:.4f}  ({hits}/{total} hits)  is_best={is_best}")
    return score, {"misses": misses_clean}
