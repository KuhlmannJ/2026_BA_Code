import argparse
import time

from pathlib import Path
import fitz  # pymupdf

import torch
from transformers import AutoModel

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
args = parser.parse_args()


#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
def select_pages(scores: torch.Tensor) -> list[int]:
    # Top-k pages by score, expanded with ±1 neighbors (Beck et al).
    n = len(scores)
    top_idx = scores.topk(min(TOP_K, n)).indices.tolist()
    pages: set[int] = set()
    for idx in top_idx:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < n:
                pages.add(neighbor)
    return sorted(pages)


def extract_pages_as_pdf(pdf_path: Path, page_indices: list[int], save_path: Path) -> None:
    src = fitz.open(str(pdf_path))
    out = fitz.open()
    
    for idx in page_indices:
        out.insert_pdf(src, from_page=idx, to_page=idx)
    
    src.close()
    save_path.write_bytes(out.tobytes())
    out.close()



if args.test :
    banner("THIS IS A TEST-RUN")
#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

# Loding and logging loaded .env
from dotenv import load_dotenv, find_dotenv
print(".env loaded:", load_dotenv(find_dotenv()))

TIME_ROUND = 6

MODEL_NAME = 'nvidia/llama-nemotron-colembed-vl-3b-v2'
ATTN_IMPL  = "flash_attention_2"
TOP_K      = 3

PDF_DIR  = Path("/scratch/tmp/jkuhlma1/data/test_esg_reports") if args.test else Path("/scratch/tmp/jkuhlma1/data/esg_reports")
PDF_LIST = sorted(list(PDF_DIR.glob("*.pdf")))

EMB_DIR  = Path("/scratch/tmp/jkuhlma1/data/embeddings/test_embeddings_colembed_3b_v2") if args.test else Path("/scratch/tmp/jkuhlma1/data/embeddings/embeddings_colembed_3b_v2")
EMD_LIST = sorted(list(EMB_DIR.glob("*.pt")))

OUTPUT_DIR     = Path("/scratch/tmp/jkuhlma1/results/A-02-answers")
RETRIEVALS_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals")

# Retrieval query — placeholder for optimize_anything / GEPA optimization
# TODO: replace with optimized query once GEPA iterations are complete
QUERY_0 = (
    "What are the total CO2 equivalent greenhouse gas emissions? "
    "Include Scope 1, Scope 2 (market-based and location-based), "
    "and Scope 3 emissions with their values, units, and reporting years."
)

# May ommit ...
#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
props      = torch.cuda.get_device_properties(0)
gpu_name   = torch.cuda.get_device_name(0)
vram_total = props.total_memory / 1e9
gpu_uuid   = props.uuid
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  UUID : {gpu_uuid}")
# May ommit ...


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
query_embeddings = model.forward_queries([QUERY_0], batch_size=1)
runtime_queryEmd = round(time.time() - t0, TIME_ROUND)
print(f"runtime_queryEmd: {runtime_queryEmd}s")


# Just O(1) for checking avail PDFs and embeddings
# And for torch.load() the embedding named the same as the PDF
pt_map = {p.stem: p for p in EMD_LIST}

for pdf_path in PDF_LIST:
    report_name = pdf_path.stem

    if report_name not in pt_map:
        print(f"[WARN] no embedding for {report_name}")
        continue

    # Step 1 — Retrieval
    print(f"Begin Loading of    {report_name}") # Spacing for printf alignment
    
    t1 = time.time()
    # image_embeddings in two steps, as loading and moving to VRAM must be done sequentially,
    # as it was saved with .cpu() to ensure cross-GPU compatibility
    image_embeddings = torch.load(pt_map[report_name], weights_only=False, map_location="cpu")
    image_embeddings = [t.to("cuda") for t in image_embeddings] # As they were saved with .cpu()
    runtime_imageEmb = round(time.time() - t1, TIME_ROUND)
    
    t2 = time.time()
    print(f"Begin Retrieval of  {report_name}")
    scores = model.get_scores(query_embeddings, image_embeddings)  # [1, n_pages]
    runtime_scoring = round(time.time() - t2, TIME_ROUND)
    
    pages  = select_pages(scores[0])
    print(f"Retreived pages 0..: {pages}")

    # Step 2 — Extract selected pages as mini-PDF
    retrieval_path = RETRIEVALS_DIR / pdf_path.name
    extract_pages_as_pdf(pdf_path, pages, retrieval_path)
    
    t3= time.time()
    runtime_PDF = round(time.time() - t3, 2)
    print(f"Mini-PDF saved:     {report_name} in {runtime_PDF}s")
    print(f"runtime_imageEmb: {runtime_imageEmb}s")
    
overall_time = round(time.time() - t0, TIME_ROUND)
print(f"DONE in {overall_time}s")