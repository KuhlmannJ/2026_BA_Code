import argparse
import time
import csv
import os
import json

from pathlib import Path
from pypdf import PdfReader, PdfWriter

import torch
from transformers import AutoModel


# Retrieval query — placeholder for optimize_anything / GEPA optimization
# Source of query text: Beck et al. (Nature Dataset)
QUERY_0 = "What are the total CO2 emissions in different years? Include Scope 1, Scope 2, and Scope 3 emissions if available."

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test",           "-t",  action="store_true",       help="Toggle Testing Path")
parser.add_argument("--gepaTrainSet",   "-gt", action="store_true",       help="Toggle Training Set of Reports")
parser.add_argument("--query",          "-q",  type=str, default=QUERY_0, help="Custom retrieval query")
args = parser.parse_args()

# Defaults to QUERY0 if none is passed on
QUERY = args.query




#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# Logging hit pages (without neighbors as they are just +- 1)
def log_pages(report_name: str, scores: torch.Tensor) -> None:
    n = len(scores)
    topk_idx = scores.topk(min(TOP_K, n)).indices.tolist()

    top10 = scores.topk(min(10, n))
    top10_pages  = top10.indices.tolist()
    top10_scores = top10.values.tolist()
    
    #csv.writer(log).writerow(["report", "phase", "top_k_pages", "timestamp", "run_ts", "top_10", "top_10_scores"])
    
    with open(RETRIEVAL_LOG, "a", newline="") as log:
        csv.writer(log).writerow([
            report_name,
            PHASE,
            topk_idx,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            RUN_TS,
            top10_pages,
            top10_scores,
        ])
  
      
# Top-k pages by score, expanded with +-1 neighbors (Beck et al).    
def select_pages(scores: torch.Tensor) -> list[int]:
    n = len(scores)
    top_idx = scores.topk(min(TOP_K, n)).indices.tolist()
    
    pages: set[int] = set()
    for idx in top_idx:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < n:
                pages.add(neighbor)
    return sorted(pages)

def extract_pages_as_pdf(pdf_path: Path, page_indices: list[int], save_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    
    for idx in page_indices:
        writer.add_page(reader.pages[idx])
    
    # = RETRIEVALS_DIR + report_name
    with open(save_path, "wb") as output_pdf:
        writer.write(output_pdf)


# Save top-10 results as JSON for GEPA optimization
# JSON can later be accessed report-wise via results["report"], results["top_10_pages"][0] etc. like a nested-array
def save_top10_results_json(report_name: str, scores: torch.Tensor) -> None:
    n = len(scores)
    top10 = scores.topk(min(10, n))
    
    results = {
        "report": report_name,
        "query": QUERY,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_ts": RUN_TS,
        "top_10_pages": top10.indices.tolist(),
        "top_10_scores": top10.values.tolist(),
        "top_10_data": [
            {"page": idx.item(), "score": score.item()} 
            for idx, score in zip(top10.indices, top10.values)
        ]
    }
    
    # Save as JSON
    json_path = RETRIEVALS_DIR / "GEPA" / f"{report_name}_top10_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)












if args.test :
    banner("THIS IS A TEST-RUN")
    
#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

# Loding and logging loaded .env
from dotenv import load_dotenv, find_dotenv
print(".env loaded:", load_dotenv(find_dotenv()))


MODEL_NAME = 'nvidia/llama-nemotron-colembed-vl-3b-v2'
# ATTN_IMPL  = "flash_attention_2" # NOT USED atm
TOP_K      = 3 # like (Beck et al)


TIME_ROUND = 6 # Rounding for time logging

# FOR CSV LOGGING OF PROGESS
PHASE           = "BECK_QUERY"
RUN_TS          = os.environ.get("RUN_TS") #Timestamp for sync evaluation from .sh file
RETRIEVAL_LOG   = Path("/scratch/tmp/jkuhlma1/results/A-02-retrieval_log.csv")
RETRIEVAL_LOG.parent.mkdir(parents=True, exist_ok=True)

if not RETRIEVAL_LOG.exists():
    with open(RETRIEVAL_LOG, "w", newline="", encoding="utf-8") as log:
        csv.writer(log).writerow(["report", "phase", "top_k_pages", "timestamp", "run_ts", "top_10", "top_10_scores"])


# The reports in the PDF_DIR dictate what Embeddings get used
if args.gepaTrainSet:
    PDF_DIR = Path("/scratch/tmp/jkuhlma1/data/training/test_esg_reports")
elif args.test:
    PDF_DIR = Path("/scratch/tmp/jkuhlma1/data/test_esg_reports")
else:
    PDF_DIR = Path("/scratch/tmp/jkuhlma1/data/esg_reports")

PDF_LIST = sorted(list(PDF_DIR.glob("*.pdf")))


# Always get all embeddings and only use those relevant for the reports in PDF_DIR
EMB_DIR = Path("/scratch/tmp/jkuhlma1/data/embeddings/embeddings_colembed_3b_v2")
EMD_LIST = sorted(list(EMB_DIR.glob("*.pt")))


# Output Path for extracted PDF Pages "Retirevals"
RETRIEVALS_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals")
RETRIEVALS_DIR.mkdir(parents=True, exist_ok=True)


print("Now used Query:")
print(QUERY)
print()
print(f"Number of PDFs inuse: {len(PDF_LIST)}")





#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
props      = torch.cuda.get_device_properties(0)
gpu_name   = torch.cuda.get_device_name(0)
vram_total = props.total_memory / 1e9
gpu_uuid   = props.uuid
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  UUID : {gpu_uuid}")


#### 2. Load Retrieval Model ####################################
banner("STEP 2: Load Retrieval Model")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    #attn_implementation=ATTN_IMPL,
).eval()

print(f" Loaded: {MODEL_NAME}")
print(f"  VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


#### 3. Begin Retrieval ####################################
banner("STEP 3: Begin Retrieval")

t0 = time.time()
query_embeddings = model.forward_queries([QUERY], batch_size=1)
runtime_queryEmd = round(time.time() - t0, TIME_ROUND)
print(f"runtime_queryEmd: {runtime_queryEmd}s")
print()

# Just O(1) for checking avail PDFs and embeddings
# And for torch.load() the embedding named the same as the PDF
pt_map = {p.stem: p for p in EMD_LIST}


for pdf_path in PDF_LIST:
    report_name = pdf_path.stem
    print(report_name)

    if report_name not in pt_map:
        print(f"[WARN] no embedding for {report_name}")
        continue

##################################################
    # Step 1 — Retrieval
    print("Begin Loading") # Spacing for printf alignment
    
    t1 = time.time()
    # image_embeddings in two steps, as loading and moving to VRAM must be done sequentially,
    # as it was saved with .cpu() to ensure cross-GPU compatibility
    
    image_embeddings = torch.load(pt_map[report_name], weights_only=False, map_location="cpu")
    image_embeddings = [t.to("cuda") for t in image_embeddings] # As they were saved with .cpu()
    
    runtime_imageEmb = round(time.time() - t1, TIME_ROUND)
    
    
    t2 = time.time()
    print("Begin Retrieval")
    
    scores = model.get_scores(query_embeddings, image_embeddings)[0]  # Pro Query eine Ziele in scores, dahe rüberall hier scores[0]
    log_pages(report_name, scores)
    save_top10_results_json(report_name, scores)  # Save Top-10 pages for GEPA optimization
    
    topk = scores.topk(min(TOP_K, len(scores)))
    print(f"Top-{TOP_K} page indices: {topk.indices.tolist()}")
    print(f"Top-{TOP_K} scores      : {topk.values.tolist()}")
    
    runtime_scoring = round(time.time() - t2, TIME_ROUND)
    
    pages  = select_pages(scores)
    print(f"Retreived pages 0..: {pages}")
    print(f" in {runtime_scoring}s")

##################################################
    # Step 2 — Extract selected pages as mini-PDF
    retrievals_path = RETRIEVALS_DIR / pdf_path.name
    extract_pages_as_pdf(pdf_path, pages, retrievals_path)
    
    print(f"Mini-PDF saved")
    print(f"runtime_imageEmb: {runtime_imageEmb}s")
    print()
    
    
    
##################################################
overall_time = round(time.time() - t0, TIME_ROUND)
print(f"DONE in {overall_time}s")