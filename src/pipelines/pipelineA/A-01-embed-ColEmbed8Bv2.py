import torch
from transformers import AutoModel

# For Logging
import argparse
import psutil
import os
import time
import sys

from pathlib import Path
import fitz  # pymupdf
from PIL import Image

# Loading API-Keys and Tokens via local .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
parser.add_argument("--batch_size", "-bz", type=int, default=2)
args = parser.parse_args()

#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


#### 0. GLOBAL VARIABLES ########################################
MODEL_NAME = 'nvidia/nemotron-colembed-vl-8b-v2'
ATTN_IMPL  = "flash_attention_2"

# Path To All and List Of All Paths to ESG-Reports
PDF_DIR  = Path("/scratch/tmp/jkuhlma1/data/esg_reports_test") if args.test else Path("/scratch/tmp/jkuhlma1/data/esg_reports")
PDF_LIST = list(PDF_DIR.glob("*.pdf"))

# Just checking ...
if not PDF_LIST:
    raise FileNotFoundError(f"Keine PDFs in {PDF_DIR}")

BATCH_SIZE = args.batch_size # 8 with ColPlali, but those embeddings will get bigger due to more vectors
DPI = 150 # matches ColEmbed's 8-tile limit (2×4 @ 512px) for A4 pages

SAVE_DIR = Path(f"/scratch/tmp/jkuhlma1/data/embeddings/{MODEL_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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
    torch.save(report_embeddings_cpu, f"{SAVE_DIR}/{report_name}.pt")
    
    # Report for each document
    pages = len(report_embeddings)
    print(f"Tensor list for {pdf_path.stem} saved. "
      f"({pages} pages | {elapsed:.1f}s | {elapsed/pages:.2f}s/page | Peak-VRAM: {peak_gb:.1f} GB | RAM: {peak_ram_gb:.1f} GB)")
    print(f"Tensor list for {report_name} saved. "
          f"({len(report_embeddings)} pages | Peak-VRAM: {peak_gb:.1f} GB)"
          f"  RAM : {peak_ram_gb:.1f} GB")
    print()
    
    
print(f"All Tensors saved to {SAVE_DIR}")

#### 5. Summary #################################################
banner("VRAM SUMMARY")
print(f"  Höchster Peak-VRAM : {global_peak_gb:.1f} GB  (Report: {global_peak_rep})")
print(f"  GPU-Kapazität      : {vram_total:.1f} GB")
print(f"  Auslastung am Peak : {100 * global_peak_gb / vram_total:.0f} %")

#################################################################
banner("DONE.")