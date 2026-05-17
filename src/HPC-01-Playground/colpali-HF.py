"""
02_colpali_test.py

ColPali v1.3 Test für PALMA II – nah am offiziellen HuggingFace-Script.

Aufruf: python 02_colpali_test.py [--pages N]
"""

import argparse
import time
import torch
from PIL import Image, ImageDraw
from dotenv import load_dotenv, find_dotenv

# ── Argumente ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pages", type=int, default=10,
                    help="Anzahl synthetischer Dokumentseiten (default: 10)")
args = parser.parse_args()

def banner(title):
    print()
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)

# Tokens laden
load_dotenv(find_dotenv())

# ── 1. GPU-Check ──────────────────────────────────────────────
banner("SCHRITT 1: GPU / CUDA")

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA nicht verfügbar! Falsche Partition?")

gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")

# Beginn vom ColPali-Entwickler-Skript
# ── 2. Modell laden ───────────────────────────────────────────
banner("SCHRITT 2: ColPali v1.3 laden")
# from typing import cast
# import torch
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

# ── 3. Synthetische Dokumentseiten ───────────────────────────
banner(f"SCHRITT 3: {args.pages} Dummy-Seiten generieren")

PAGE_TEXTS = [
    "Annual Report 2023 – Overview of financial performance",
    "Corporate governance and board structure",
    "Risk management framework and internal controls",
    "Scope 1 emissions: 12,450 tCO2e direct combustion sources",  # ← Treffer Q1
    "Supply chain management and procurement strategy",
    "Employee diversity and inclusion initiatives",           # ← Treffer Q2
    "Water consumption and waste management",
    "Social impact and community engagement programs",
    "Executive remuneration and compensation policy",
    "Independent auditor's report and assurance statement",
]

texts = [PAGE_TEXTS[i % len(PAGE_TEXTS)] for i in range(args.pages)]

def make_page_image(text: str, page_num: int) -> Image.Image:
    img  = Image.new("RGB", (595, 842), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((500, 20), f"p. {page_num}", fill=(150, 150, 150))
    draw.line([(40, 50), (555, 50)], fill=(200, 200, 200), width=1)
    draw.text((40, 80), text, fill=(30, 30, 30))
    for i, y in enumerate(range(130, 700, 22)):
        gray = 180 + (i % 3) * 20
        draw.line([(40, y), (400 + (i * 13) % 155, y)],
                  fill=(gray, gray, gray), width=2)
    return img

# Your inputs (COLPALI)
images = [make_page_image(t, i + 1) for i, t in enumerate(texts)]
for i, t in enumerate(texts):
    print(f"  Seite {i+1:2d}: {t[:60]}")

queries = [
    "What are the Scope 1 CO2 emissions of the company?",
    "What are the diversity and inclusion initiatives?",
]

# ── 4. Forward Pass (offizielles Pattern) ────────────────────
banner("SCHRITT 4: Embeddings berechnen (GPU-Workload)")

t0 = time.time()

# Process the inputs (COLPALI)
batch_images   = processor.process_images(images).to(model.device)
batch_queries  = processor.process_queries(queries).to(model.device)

# Forward pass (COLAPLI)
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

elapsed = time.time() - t0
print(f"  ✅ Fertig in {elapsed:.1f}s")
print(f"  image_embeddings : {image_embeddings.shape}")
print(f"  query_embeddings : {query_embeddings.shape}")
print(f"  VRAM belegt      : {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ── 5. Scoring & Ranking ─────────────────────────────────────
banner("SCHRITT 5: Scoring & Ranking")

# scores: (n_queries, n_pages)
scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(f"  Score-Matrix: {scores.shape}  (queries × pages)\n")

expected = ["Scope 1", "diversity"]
all_correct = True

for q_idx, query in enumerate(queries):
    print(f"  Query: {query!r}")
    ranked = scores[q_idx].argsort(descending=True)
    for rank, page_idx in enumerate(ranked[:5]):
        page_idx = page_idx.item()
        score    = scores[q_idx][page_idx].item()
        marker   = " ← 🎯" if expected[q_idx] in texts[page_idx] else ""
        print(f"    #{rank+1}  Seite {page_idx+1:2d} | Score {score:7.2f} | "
              f"{texts[page_idx][:45]}{marker}")
    best = ranked[0].item()
    if expected[q_idx] not in texts[best]:
        all_correct = False
    print()

# ── Zusammenfassung ───────────────────────────────────────────
banner("ZUSAMMENFASSUNG")
print(f"  Seiten verarbeitet : {len(images)}")
print(f"  GPU                : {gpu_name}")
print(f"  VRAM gesamt        : {vram_total:.1f} GB")
print(f"  Laufzeit           : {elapsed:.1f}s ({elapsed/len(images)*1000:.0f} ms/Seite)")
print(f"  Retrieval korrekt  : {'✅ Ja' if all_correct else '❌ Nein'}")
print()
print("  🎉 Test erfolgreich abgeschlossen!")