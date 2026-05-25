#### 0. Prework #################################################
import time
import os
import torch
import fitz  # pymupdf

from pathlib import Path
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from colpali_engine.models import ColPali, ColPaliProcessor

# Load Tokens
load_dotenv(find_dotenv)

# Path to and List of ESG-Reports
pdf_dir  = Path("$WORK/data/esg_reports")
pdf_list = list(pdf_dir.glob("*.pdf"))

# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


#### 1. GPU Details #############################################
banner("SCHRITT 1: GPU / CUDA")
gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")

#### 2. Load ColPali ############################################
model_name = "vidore/colpali-v1.3"

t0 = time.time()

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

print(f"  Geladen in {time.time() - t0:.1f}s")
print(f"  VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

#### 3. PDF ############################################

## WIP: Ziel, denn ich brauch nur das

# Process & embed
batch_images = processor.process_images(images).to(model.device)
with torch.no_grad():
    image_embeddings = model(**batch_images)

# Speichern
torch.save(image_embeddings.cpu(), save_path)