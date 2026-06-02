# Is in colembed script, but why?
from io import BytesIO

import torch
from transformers import AutoModel

import argparse
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

# Loading API-Keys and Tokens via local .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Toggle Testing Path")
args = parser.parse_args()

#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


#### 0. GLOBAL VARIABLES ########################################
MODEL_NAME = 'nvidia/llama-nemotron-colembed-vl-3b-v2'
ATTN_IMPL  = "flash_attention_2"

# Path To All and List Of All Paths to ESG-Reports
PDF_DIR  = Path("/scratch/tmp/jkuhlma1/data/esg_reports_test") if args.t else Path("/scratch/tmp/jkuhlma1/data/esg_reports")
PDF_LIST = list(PDF_DIR.glob("*.pdf"))

# Just checking ...
if not PDF_LIST:
    raise FileNotFoundError(f"Keine PDFs in {PDF_DIR}")

BATCH_SIZE = 2 # 8 with ColPlali, but those embeddings will get bigger due to more vectors
DPI = 150 # matches ColEmbed's 8-tile limit (2×4 @ 512px) for A4 pages

SAVE_DIR = Path("/scratch/tmp/jkuhlma1/data/embeddings/embeddings_colembed_3b_v2")


# May ommit ...
#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
# May ommit ...


#### 2. Load Retrieval Model ####################################
model = AutoModel.from_pretrained(
    MODEL_NAME,
    device_map='cuda:0',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation=ATTN_IMPL
).eval()

print(f"  VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


#### 3. PDF to Image ############################################
banner("STEP 3: PDF to Image")
# => Das würde jetzt erstmal alle PDFs in Bilder umwandeln ...
# Man könnte das natürlich auslagern und nur genau das PDF umwandeln, was man grad braucht ...

# Will be a compelte key-value dict of report names and images of the reports
pdf_img_dict = {}

for pdf_path in PDF_LIST :
    fitz.TOOLS.reset_mupdf_warnings()  # Clear Buffer
    
    current_pdf_imgages = []
    
    with fitz.open(str(pdf_path)) as doc :
        
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDf is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            current_pdf_imgages.append(img)
    
    # Need to do more error handling
    warnings = fitz.TOOLS.mupdf_warnings()
    if warnings:
        print(f"  [WARN] {pdf_path.name}: {warnings}")
    # More logging       
    print(f"{pdf_path.stem} vollständig verarbeitet.")
            
    ### HIER könnte jetzt auch 4. einsetzen ...
    
    pdf_img_dict[pdf_path.stem] = current_pdf_imgages
    
#### 4. Create Embeddings #######################################
banner("STEP 4: Create Embeddings")

# Global VRAM Peak over all reports
global_peak_gb  = 0.0
global_peak_rep = None
    
# Iterating trough all PDFs in pdf_img_dict
for report_name, report_imgs in pdf_img_dict.items():
    
    torch.cuda.reset_peak_memory_stats()
    
    # No more intermidiate step necessary due to internal batching of ColEmbed3B-v2


    # Forward pass – compute embeddings for images of batch
    ### HEAVY COMPUTE ###
    with torch.no_grad():
        report_embeddings = model.forward_images(report_imgs, batch_size=BATCH_SIZE)
        
    # Peak of this report
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    if peak_gb > global_peak_gb:
        global_peak_gb  = peak_gb
        global_peak_rep = report_name
        
    # Pro Seite auf CPU lösen und als Liste persistieren (Seitengrenzen bleiben erhalten)
    report_embeddings_cpu = [emb.detach().cpu() for emb in report_embeddings]
    
    ## Saving every report tensor seperately
    torch.save(report_embeddings_cpu, f"{SAVE_DIR}/{report_name}.pt")
    
    # VRAM-Peak for each report
    print(f"Tensor list for {report_name} saved. "
          f"({len(report_embeddings)} pages | Peak-VRAM: {peak_gb:.1f} GB)")

print(f"All Tensors saved to {SAVE_DIR}")

#### 5. Summary #################################################
banner("VRAM SUMMARY")
print(f"  Höchster Peak-VRAM : {global_peak_gb:.1f} GB  (Report: {global_peak_rep})")
print(f"  GPU-Kapazität      : {vram_total:.1f} GB")
print(f"  Auslastung am Peak : {100 * global_peak_gb / vram_total:.0f} %")

#################################################################
banner("DONE.")