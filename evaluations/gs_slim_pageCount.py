import os
import csv
import PyPDF2
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))

esg_all = Path(BASE) / "../localdata/esg_reports"
output_csv = Path(BASE) / "gs_slim_pageCount.csv"

results = []

for root, dirs, files in os.walk(esg_all):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = len(reader.pages)
                results.append((filename, pages))
            except Exception as e:
                print(f"Fehler bei {filename}: {e}")
                results.append((filename, None))

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "pages"])
    writer.writerows(results)

print(f"Fertig. {len(results)} PDFs verarbeitet. Gespeichert unter: {output_csv}")