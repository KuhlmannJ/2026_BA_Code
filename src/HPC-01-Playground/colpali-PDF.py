"""
03_colpali_esg.py

ColPali v1.3 Inferenz auf einem echten ESG-Report (PDF).

Aufruf:
    python 03_colpali_esg.py --pdf /path/to/report.pdf --query "Scope 1 emissions"

Optional:
    --dpi       Auflösung beim PDF-Rendering (default: 150)
    --top_k     Wie viele Top-Seiten ausgeben (default: 5)
    --save      Embeddings speichern unter $WORK/data/embeddings/
"""

import argparse
import time
import os
from pathlib import Path

from PIL import Image

import torch
import fitz  # pymupdf
from dotenv import load_dotenv, find_dotenv

# ── Argumente ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pdf",    required=True,  help="Pfad zur PDF-Datei")
parser.add_argument("--query",  required=True,  help="Suchanfrage")
parser.add_argument("--dpi",    type=int, default=150)
parser.add_argument("--top_k",  type=int, default=5)
parser.add_argument("--save",   action="store_true", help="Embeddings speichern")
args = parser.parse_args()

def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

# Tokens laden
load_dotenv(find_dotenv())

# ── 1. GPU-Check ──────────────────────────────────────────────
banner("SCHRITT 1: GPU / CUDA")

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA nicht verfügbar!")

gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")

# ── 2. Modell laden ───────────────────────────────────────────
banner("SCHRITT 2: ColPali v1.3 laden")

from colpali_engine.models import ColPali, ColPaliProcessor

model_name = "vidore/colpali-v1.3"
print(f"  Lade {model_name} ...")
# Bei Skripten unbedingt im Job-Script angeben!
# export HF_HOME=$WORK/cache/huggingface
print("  (Erster Download ~5 GB – danach gecacht in $WORK/cache/huggingface)")

t0 = time.time()
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)
print(f"  ✅ Geladen in {time.time() - t0:.1f}s")
print(f"  VRAM belegt: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ── 3. PDF → Bilder ───────────────────────────────────────────
banner("SCHRITT 3: PDF einlesen")

pdf_path = Path(args.pdf)
print(f"  Datei : {pdf_path.name}")
print(f"  DPI   : {args.dpi}")

# Your inputs (COLPALI)
t0 = time.time()
doc = fitz.open(str(pdf_path))
images = []
for page in doc:
    pix = page.get_pixmap(dpi=args.dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)
doc.close()
print(f"  ✅ {len(images)} Seiten eingelesen in {time.time() - t0:.1f}s")

# ── 4. Embeddings berechnen ───────────────────────────────────
banner("SCHRITT 4: Seiten embedden")

# Process the inputs (COLPALI)
t0 = time.time()
batch_images  = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries([args.query]).to(model.device)

# Forward pass (COLAPLI)
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

elapsed = time.time() - t0
print(f"  ✅ {len(images)} Seiten embedded in {elapsed:.1f}s ({elapsed/len(images)*1000:.0f} ms/Seite)")
print(f"  VRAM belegt: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ── 5. Embeddings speichern (optional) ───────────────────────
if args.save:
    banner("SCHRITT 5: Embeddings speichern")
    work = os.environ.get("WORK", str(Path.home()))
    save_dir = Path(work) / "data" / "embeddings"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{pdf_path.stem}.pt"
    torch.save(image_embeddings.cpu(), save_path)
    print(f"  ✅ Gespeichert unter: {save_path}")

# ── 6. Scoring & Ranking ──────────────────────────────────────
banner("SCHRITT 6: Retrieval")

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
ranked = scores[0].argsort(descending=True)

print(f"  Query : {args.query!r}")
print(f"  Top {args.top_k} Seiten:\n")
for rank, page_idx in enumerate(ranked[:args.top_k]):
    page_idx = page_idx.item()
    score    = scores[0][page_idx].item()
    print(f"  #{rank+1}  Seite {page_idx+1:3d}  |  Score {score:7.2f}")

# ── Zusammenfassung ───────────────────────────────────────────
banner("ZUSAMMENFASSUNG")
print(f"  Report    : {pdf_path.name}")
print(f"  Seiten    : {len(images)}")
print(f"  Query     : {args.query!r}")
print(f"  Laufzeit  : {elapsed:.1f}s ({elapsed/len(images)*1000:.0f} ms/Seite)")
print(f"  GPU       : {gpu_name}")
print(f"  VRAM ges. : {vram_total:.1f} GB")