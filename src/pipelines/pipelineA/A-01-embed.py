import torch
from transformers import AutoModel

# For Logging
import argparse
import psutil
import os
import time
import sys
import csv
from datetime import datetime

from pathlib import Path
import fitz  # pymupdf
from PIL import Image

# Loading API-Keys and Tokens via local .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
parser.add_argument("--all",  "-a", action="store_true", help="Toggle ALL ESG reports (useless aswell)") # TAKES LONGER
parser.add_argument("--batch_size", "-bz", type=int, default=8) #8 seems fastest overall due to steady workload of GPU

# ── Arguments MODEL_SELECTION # 'dest' for numbers in flags
parser.add_argument("-3B", dest="_3B", action="store_true", help="nvidia/llama-nemotron-colembed-vl-3b-v2")
parser.add_argument("-4B", dest="_4B", action="store_true", help="nvidia/nemotron-colembed-vl-4b-v2")
parser.add_argument("-8B", dest="_8B", action="store_true", help="nvidia/nemotron-colembed-vl-8b-v2")
args = parser.parse_args()

#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

banner("START: A-01-embed.py")
#### 0. GLOBAL VARIABLES ########################################

SCRATCH_ROOT = Path("/scratch/tmp/jkuhlma1")

match True:
    case args._3B:
        MODEL_NAME = "nvidia/llama-nemotron-colembed-vl-3b-v2"
    case args._4B:
        MODEL_NAME = "nvidia/nemotron-colembed-vl-4b-v2"
    case args._8B:
        MODEL_NAME = "nvidia/nemotron-colembed-vl-8b-v2"
    case _:
        parser.error("Set a MODEL_NAME flag '-3B' or '-4B' or '-8B'.") # No default chosen, as not necessary with .sh 

if args._3B :
    ATTN_IMPL  = None # Does not work on 3B
else :
    ATTN_IMPL = "flash_attention_2"

# Path To All and List Of All Paths to ESG-Reports

match True:
    case args.test:
        PDF_DIR  = SCRATCH_ROOT / "data" / "esg_reports_test"
    case args.all:
        PDF_DIR  = SCRATCH_ROOT / "data" / "all_esg_reports"
    case _:
        PDF_DIR  = SCRATCH_ROOT / "data" / "esg_reports"

PDF_LIST = list(PDF_DIR.glob("*.pdf"))

# Just checking ...
if not PDF_LIST:
    raise FileNotFoundError(f"Keine PDFs in {PDF_DIR}")

BATCH_SIZE = args.batch_size # 8 with ColPlali, but those embeddings will get bigger due to more vectors
DPI = 150 # matches ColEmbed's 8-tile limit (2×4 @ 512px) for A4 pages

SAVE_DIR = (SCRATCH_ROOT / "data" / "embeddings" / "all" / MODEL_NAME) if args.all else (SCRATCH_ROOT / "data" / "embeddings" / MODEL_NAME)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SAVE_DIR / f"kpi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_FIELDS = ["report", "pages", "elapsed_s", "s_per_page", "peak_vram_gb", "peak_ram_gb", "file_size_mb", "embed_dim"]
with open(LOG_FILE, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

#### PARAMS OUTPUT
banner("STEP 0: PARAMS")
print(f"MODEL_NAME : {MODEL_NAME}")
print(f"PDF_DIR    : {PDF_DIR}")
print(f"No. of PDF : {len(PDF_LIST)}")
print(f"BATCH_SIZE : {BATCH_SIZE}")
print(f"KPI-Log    : {LOG_FILE}")
print()
print("=" * 60)
print()


#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
props      = torch.cuda.get_device_properties(0)
gpu_name   = torch.cuda.get_device_name(0)
vram_total = props.total_memory / 1e9
gpu_uuid   = props.uuid
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  UUID : {gpu_uuid}")




if args.test :
    banner("THIS IS A TEST-RUN")
    
if args.all :
    banner("ALL IS SELECTED. HUGE DATASET.")
#### 2. Load Retrieval Model ####################################
banner("STEP 2: Load Retrieval Model")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    device_map='cuda:0',
    trust_remote_code=True,
    dtype=torch.bfloat16, # torch_dtype → dtype => "torch_dtype` is deprecated" 
    attn_implementation=ATTN_IMPL # TODO NEEDS A RE-RUN, was disabled at last run
).eval()
print(f" Loaded: {MODEL_NAME}")
print(f" Attention loaded:{model.config._attn_implementation}")
print(f" VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


#### 3. PDF to Image direct into Embedding ############################################
banner("STEP 3: PDF to Image to Embedding")

process = psutil.Process(os.getpid())

# Global VRAM Peak over all reports
global_peak_gb  = 0.0
global_peak_rep = None

for pdf_path in PDF_LIST :
    fitz.TOOLS.reset_mupdf_warnings()  # Clear Buffer
    
    report_name = pdf_path.stem
    current_pdf_imgages = []
    
    with fitz.open(str(pdf_path)) as doc :
        
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDf is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            current_pdf_imgages.append(img)
    
    # Need to do more error handling
    warnings = fitz.TOOLS.mupdf_warnings()
    if warnings:
        print(f"  [WARN] {pdf_path.name}: {warnings}", file=sys.stderr)
    # More logging       
    print(f"Complete: From PDF 2 Image {report_name}.")
            
    
    ##### Embedding #######################
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    
    with torch.no_grad():
        report_embeddings = model.forward_images(current_pdf_imgages, batch_size=BATCH_SIZE)
        
    elapsed = time.time() - t0
    
    peak_ram_gb = process.memory_info().rss / 1e9
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    
    # Dump from RAM
    del current_pdf_imgages
    
    if peak_gb > global_peak_gb:
        global_peak_gb  = peak_gb
        global_peak_rep = report_name
    
    # Detaching images page-by-page from CPU and 'save' as list. Keeps pagenumbers intact.
    report_embeddings_cpu = [emb.detach().cpu() for emb in report_embeddings]
    
    ## Saving every report tensor seperately
    pt_path = SAVE_DIR / f"{report_name}.pt"
    torch.save(report_embeddings_cpu, pt_path)

    # Report for each document
    pages      = len(report_embeddings)
    file_mb    = pt_path.stat().st_size / 1e6
    embed_dim  = report_embeddings_cpu[0].shape[-1] if report_embeddings_cpu else None

    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow({
            "report":       report_name,
            "pages":        pages,
            "elapsed_s":    round(elapsed, 2),
            "s_per_page":   round(elapsed / pages, 3),
            "peak_vram_gb": round(peak_gb, 2),
            "peak_ram_gb":  round(peak_ram_gb, 2),
            "file_size_mb": round(file_mb, 2),
            "embed_dim":    embed_dim,
        })

    print(f"Tensor list for {report_name} saved. "
          f"({pages} pages | {elapsed:.1f}s | {elapsed/pages:.2f}s/page | "
          f"Peak-VRAM: {peak_gb:.1f} GB | RAM: {peak_ram_gb:.1f} GB | {file_mb:.1f} MB)")
    print()
    
    
print(f"All Tensors saved to {SAVE_DIR}")
print(f"KPI-Log written to  {LOG_FILE}")

#### 5. Summary #################################################
banner("VRAM SUMMARY")
print(f"  Höchster Peak-VRAM : {global_peak_gb:.1f} GB  (Report: {global_peak_rep})")
print(f"  GPU-Kapazität      : {vram_total:.1f} GB")
print(f"  Auslastung am Peak : {100 * global_peak_gb / vram_total:.0f} %")

#################################################################
banner("END: A-01-embed.py")