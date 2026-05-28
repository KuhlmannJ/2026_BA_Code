import time
import os
import argparse
import torch
import fitz  # pymupdf

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

# Loading API-Keys and Tokens via local .env
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

MODEL_NAME = "vidore/colpali-v1.3"

# Path To All and List Of All Paths to ESG-Reports
PDF_DIR  = Path("/scratch/tmp/jkuhlma1/data/esg_reports_test") if args.t else Path("/scratch/tmp/jkuhlma1/data/esg_reports")
PDF_LIST = list(PDF_DIR.glob("*.pdf"))

# Just checking ...
if not PDF_LIST:
    raise FileNotFoundError(f"Keine PDFs in {PDF_DIR}")

BATCH_SIZE = 8 #8 seems to work well on H200mini nodes
DPI = 150 # Often chosen size for vector-based PDF

SAVE_DIR = Path("/scratch/tmp/jkuhlma1/data/embeddings")

# May ommit ...
#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
# May ommit ...


#### 2. Load ColPali ############################################
banner("STEP 2: Load ColPali")
model = ColPali.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

print(f"  VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


#### 3. PDF to Image ############################################
banner("STEP 3: PDF to Image")
# => Das würde jetzt erstmal alle PDFs in Bilder umwandeln ...
# Man könnte das natürlich auslagern und nur genau das PDF umwandeln, was man grad braucht ...

# Will be a compelte key-value dict of report names and images of the reports
pdf_img_dict = {}

for pdf_path in PDF_LIST :
    
    current_pdf_imgages = []
    
    with fitz.open(str(pdf_path)) as doc :
        
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDf is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            current_pdf_imgages.append(img)
    
    # Mehr logging       
    print(f"{pdf_path} vollständig verarbeitet")
            
    ### HIER könnte jetzt auch 4. einsetzen ...
    
    pdf_img_dict[pdf_path.stem] = current_pdf_imgages


#### 4. Create Embeddings #######################################
banner("STEP 4: Create Embeddings")

# Iterating trough all PDFs in pdf_img_dict
for report_name, report_imgs in pdf_img_dict.items():
    
    ## This intermediate step is necessary to not overlad VRAM with all pages at once
    report_embeddings = []
    
    # Splitting PDF into BATCH_SIZE sized batches
    for img in range(0, len(report_imgs), BATCH_SIZE):
        
        # The actual images of the pages in that batch to be made in to tensors
        batch = report_imgs[img : img + BATCH_SIZE]
        
        # Pre-Processing inputs → tensors
        batch_input = processor.process_images(batch).to(model.device)

        #Forward pass – compute embeddings for images of batch
        ### HEAVY COMPUTE ###
        with torch.no_grad():
            batch_embeddings = model(**batch_input)
        
        report_embeddings.append(batch_embeddings.cpu())
    
    ## Saving every report tensor seperately
    report_tensor = torch.cat(report_embeddings, dim=0)
    torch.save(report_tensor.cpu(), f"{SAVE_DIR}/{report_name}.pt") #.cpu() redundant, but clear that everything is unbounded from GPU
    
    print(f"Tensor for {report_name} saved.")

print(f"All Tensors saved to {SAVE_DIR}")
banner("DONE.")