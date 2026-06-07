import ast
import csv
from pathlib import Path
import os

import pandas as pd

#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

# ── Adjust these paths ───────────────────────────────────────
GOLD_PATH      = Path("/scratch/tmp/jkuhlma1/data/checklist.csv")
RETRIEVAL_LOG  = Path("/scratch/tmp/jkuhlma1/results/A-02-retrieval_log.csv")
OUTPUT_DIR     = Path("/scratch/tmp/jkuhlma1/evaluations/A-02")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS_FILTER = os.environ.get("RUN_TS") # From sh script for concanated evaluation
# ─────────────────────────────────────────────────────────────

# NOTE: Gold standard pages are printed page numbers (as shown in PDF viewer).
# Retrieval log pages are 0-indexed PyMuPDF indices.
# PDFs starting at printed page 1 (normal) have offset -1 (gold 63 = index 62).
# PDFs with non-numeric first pages (e.g. Allianz "A") have offset 0 (gold 78 = index 78).
# We check both gold_page and gold_page-1 against retrieved indices to handle both cases.


#### 1. Load Gold Standard ######################################
banner("STEP 1: Load Gold Standard")

gold = pd.read_csv(GOLD_PATH, dtype={"page": str})  # page as str: handles "Env33" etc.
gold["report_stem"] = gold["report_name"].str.replace(r"\.pdf$", "", regex=True)

# Get unique (report, page) pairs — we only care about whether the page was found,
# not how many scope/year entries are on it
gold_pages = (
    gold[["report_stem", "page"]]
    .drop_duplicates()
    .copy()
)

# Keep only numeric pages for retrieval matching
gold_pages["page_int"] = pd.to_numeric(gold_pages["page"], errors="coerce")
gold_pages_numeric = gold_pages.dropna(subset=["page_int"]).copy()
gold_pages_numeric["page_int"] = gold_pages_numeric["page_int"].astype(int)

print(f"  Reports:          {gold['report_stem'].nunique()}")
print(f"  Non-numeric pages (skipped for retrieval): {gold_pages['page_int'].isna().sum()}")


#### 2. Load Retrieval Log ######################################
banner("STEP 2: Load Retrieval Log")

retrieval_rows = []
with open(RETRIEVAL_LOG, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if RUN_TS_FILTER and row["run_ts"] != RUN_TS_FILTER:
            continue
        top_k  = ast.literal_eval(row["top_k_pages"])
        # Expand with ±1 neighbors (Beck et al.) to get the full retrieved set
        expanded = set()
        for idx in top_k:
            for neighbor in (idx - 1, idx, idx + 1):
                if neighbor >= 0:
                    expanded.add(neighbor)
        retrieval_rows.append({
            "report_stem": row["report"],
            "phase":       row["phase"],
            "top_k_pages": top_k,
            "expanded":    expanded,
        })

ret_df = pd.DataFrame(retrieval_rows)
print(f"  Retrieval log entries: {len(ret_df)}")
print(f"  Reports in log:        {ret_df['report_stem'].nunique()}")


#### 3. Retrieval Evaluation ####################################
banner("STEP 3: Retrieval Evaluation")

merged = gold_pages_numeric.merge(ret_df, on="report_stem", how="inner")

def hit_topk(row):
    # Check gold_page and gold_page-1 (handles ±1 offset between PDFs)
    return row["page_int"] in row["top_k_pages"] or (row["page_int"] - 1) in row["top_k_pages"]

def hit_expanded(row):
    return row["page_int"] in row["expanded"] or (row["page_int"] - 1) in row["expanded"]

merged["hit_topk"]     = merged.apply(hit_topk,     axis=1)
merged["hit_expanded"] = merged.apply(hit_expanded, axis=1)

n_total    = len(merged)
n_topk     = merged["hit_topk"].sum()
n_expanded = merged["hit_expanded"].sum()

print(f"  Evaluated: {n_total} (report, page) pairs across {merged['report_stem'].nunique()} reports")
print()
print(f"  Hit@TopK     (before ±1 expansion): {n_topk/n_total:.1%}  ({n_topk}/{n_total})")
print(f"  Hit@Expanded (after  ±1 expansion): {n_expanded/n_total:.1%}  ({n_expanded}/{n_total})")


#### 4. Per-Report Breakdown ####################################
banner("STEP 4: Per-Report Breakdown (just to file)")

per_report = (
    merged.groupby("report_stem")
    .agg(
        gold_pages    = ("page_int",     "count"),
        hit_topk      = ("hit_topk",     "sum"),
        hit_expanded  = ("hit_expanded", "sum"),
    )
    .reset_index()
)
per_report["hit_topk_pct"]    = per_report["hit_topk"]    / per_report["gold_pages"]
per_report["hit_expanded_pct"] = per_report["hit_expanded"] / per_report["gold_pages"]

# Flag reports with complete miss (no gold page retrieved at all)
per_report["full_miss"] = per_report["hit_expanded"] == 0

full_misses = per_report[per_report["full_miss"]]["report_stem"].tolist()
print(f"\n  Full misses ({len(full_misses)} reports): {full_misses}")


#### 5. Save ####################################################
banner("STEP 5: Save")

merged.drop(columns=["expanded"]).to_csv(
    OUTPUT_DIR / "retrieval_evaluation.csv", index=False
)
per_report.to_csv(OUTPUT_DIR / "retrieval_per_report.csv", index=False)

print(f"  → {OUTPUT_DIR / 'retrieval_evaluation.csv'}")
print(f"  → {OUTPUT_DIR / 'retrieval_per_report.csv'}")