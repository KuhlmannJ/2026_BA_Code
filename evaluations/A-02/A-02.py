import ast
import csv
import json
from pathlib import Path
import os
import argparse

import pandas as pd

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--local", "-l", action="store_true", help="For local execution and Paths")
parser.add_argument("--runts", "-r", default="0613_1445", help="For local execution a given RUN_TS")
args = parser.parse_args()

#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

# ── Adjust these paths ───────────────────────────────────────

if args.local :
    GOLD_PATH      = Path(__file__).parent.parent / "gs_slim.json"
    RETRIEVAL_LOG  = Path("../../localdata/A-02-retrieval_log.csv")
    OUTPUT_DIR     = Path("../A-02")
    RUN_TS         = args.runts
else:
    GOLD_PATH      = Path("/home/j/jkuhlma1/2026_BA_Code/evaluations/gs_slim.json")
    RETRIEVAL_LOG  = Path("/scratch/tmp/jkuhlma1/results/A-02-retrieval_log.csv")
    OUTPUT_DIR     = Path("/scratch/tmp/jkuhlma1/evaluations/A-02")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_TS = os.environ.get("RUN_TS") # From sh script for concanated evaluation

#### PARAMS OUTPUT
banner("STEP 0: PARAMS")
print(f"GOLD_PATH     : {GOLD_PATH}")
print(f"RETRIEVAL_LOG : {RETRIEVAL_LOG}")
print(f"OUTPUT_DIR    : {OUTPUT_DIR}")
print(f"RUN_TS        : {RUN_TS}")
print("=" * 60)
print()
# ─────────────────────────────────────────────────────────────

# NOTE: Gold standard pages are printed page numbers (as shown in PDF viewer).
# Retrieval log pages are 0-indexed PyMuPDF indices.
# PDFs starting at printed page 1 (normal) have offset -1 (gold 63 = index 62).
# PDFs with non-numeric first pages (e.g. Allianz "A") have offset 0 (gold 78 = index 78).
# We check both gold_page and gold_page-1 against retrieved indices to handle both cases.
# gs_slim.json is used as gold source (pages already cleaned to plain integer strings).


#### 1. Load Gold Standard ######################################
banner("STEP 1: Load Gold Standard")

with open(GOLD_PATH, encoding="utf-8") as f:
    gs = pd.DataFrame(json.load(f))

# NOTE: Get unique (report, page) pairs. We only care about whether the page was found,
# not how many scope/year entries are on it
gold_pages = (
    gs[gs["page"].notna()][["report_name", "page"]]
    .drop_duplicates()
    .copy()
)

print(f"Reports: {gs['report_name'].nunique()}")
print(f"RUN_TS: {RUN_TS}")


#### 2. Load Retrieval Log ######################################
banner("STEP 2: Load Retrieval Log")

RETRIEVAL_LOG_rows = []
with open(RETRIEVAL_LOG, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        
        # Skips older rows
        if row["run_ts"] != RUN_TS:
            continue
        
        top_k  = ast.literal_eval(row["top_k_pages"])
        
        # Expand with ±1 neighbors (Beck et al.) to get the full retrieved pages
        expanded = []
        for idx in top_k:
            for neighbor in (idx - 1, idx, idx + 1):
                if neighbor >= 0:
                    expanded.append(neighbor)
                    
        RETRIEVAL_LOG_rows.append({
            "model":         row["model"],
            "report_name":   row["report"],
            "phase":         row["phase"],
            "top_k_pages":   top_k,
            "expanded":      expanded,
            "top_10":        row["top_10"],
            "top_10_scores": row["top_10_scores"]
        })

ret_df = pd.DataFrame(RETRIEVAL_LOG_rows)
print(f"Retrieval log entries: {len(ret_df)}")
print(f"Reports in log:        {ret_df['report_name'].nunique()}")
print(f"Retrieval Model used:  {ret_df["model"][0]}")


### MISSING REPORTS?
# Compare report sets: which reports are present in one but not the other
ret_reports = set(ret_df['report_name'].unique())
gold_reports = set(gs['report_name'].unique())

in_gold_not_ret = sorted(gold_reports - ret_reports)
in_ret_not_gold = sorted(ret_reports - gold_reports)

print(f"Reports only in gold ({len(in_gold_not_ret)}): {in_gold_not_ret}")
print(f"Reports only in retrieval log ({len(in_ret_not_gold)}): {in_ret_not_gold}")

#### 3. Retrieval Evaluation ####################################
banner("STEP 3: Retrieval Evaluation")

merged = gold_pages.merge(ret_df, on="report_name", how="left")

def hit_topk(row):
    top_k = row["top_k_pages"]
    if not isinstance(top_k, (list, set)):
        return False
    try:
        page = int(row["page"])
    except (ValueError, TypeError):
        return False
    return page in top_k or (page - 1) in top_k

def hit_expanded(row):
    expanded = row["expanded"]
    if not isinstance(expanded, (list, set)):
        return False
    try:
        page = int(row["page"])
    except (ValueError, TypeError):
        return False
    return page in expanded or (page - 1) in expanded

merged["hit_topk"]     = merged.apply(hit_topk,     axis=1)
merged["hit_expanded"] = merged.apply(hit_expanded, axis=1)

n_total    = len(merged)
n_topk     = merged["hit_topk"].sum()
n_expanded = merged["hit_expanded"].sum()

# ── MRR@3: rank of first hit within top-3 only ──────────────────
merged["top_10_list"] = merged["top_10"].apply(ast.literal_eval)

def first_hit_rank_top3(row):
    try:
        page = int(row["page"])
    except (ValueError, TypeError):
        return None
    for i, idx in enumerate(row["top_10_list"][:3]):
        if idx == page or idx == page - 1:
            return i + 1
    return None

merged["rank"] = merged.apply(first_hit_rank_top3, axis=1)
mrr = merged["rank"].apply(lambda r: 1 / r if r else 0).mean()

print(f"  Evaluated: {n_total} (report, page) pairs across {merged['report_name'].nunique()} reports")
print()
print(f"  Recall@3 (before ±1 expansion): {n_topk/n_total:.1%}  ({n_topk}/{n_total})")
print(f"  Recall@3 (after  ±1 expansion): {n_expanded/n_total:.1%}  ({n_expanded}/{n_total})")
print(f"  MRR@3                         : {mrr:.3f}")
print()
print("  Note: offset-correction (page and page-1) applied in all metrics.")
print("  ±1 neighbour expansion is a separate generosity layer (Beck et al.).")

per_report = (
    merged.groupby("report_name")
    .agg(
        gold_pages    = ("page",         "count"),
        hit_topk      = ("hit_topk",     "sum"),
        hit_expanded  = ("hit_expanded", "sum"),
    )
    .reset_index()
)
# Flag reports with complete miss (no gold page retrieved at all)
per_report["full_miss"] = per_report["hit_expanded"] == 0

full_misses = per_report[per_report["full_miss"]]["report_name"].tolist()
print(f"\n  Full misses ({len(full_misses)} reports): {full_misses}")

# ── Missed (report, page) pairs after ±1 expansion ──────────────
misses = merged[~merged["hit_expanded"]][["report_name", "page", "phase", "top_k_pages", "top_10"]].copy()
misses = misses.sort_values(["report_name", "page"])

print(f"\n  Missed pages ({len(misses)} total):")
for _, row in misses.iterrows():
    print(f"    {row['report_name']}  p.{row['page']}  (top_k={row['top_k_pages']})")


#### 4. Save ####################################################
banner("STEP 4: Save")

merged_sorted = merged[["model","report_name","page","phase","top_k_pages","hit_topk","hit_expanded","top_10","top_10_scores"]]
merged_sorted.to_csv(OUTPUT_DIR / "retrieval_evaluation.csv", index=False)
print(f"  → {OUTPUT_DIR / 'retrieval_evaluation.csv'}")

misses.to_csv(OUTPUT_DIR / "retrieval_misses.csv", index=False)
print(f"  → {OUTPUT_DIR / 'retrieval_misses.csv'}")