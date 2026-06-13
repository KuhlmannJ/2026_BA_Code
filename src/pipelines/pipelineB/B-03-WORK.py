import argparse

import base64
import io
import json

import fitz #pip install pymupdf
from PIL import Image
from pathlib import Path

from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
args = parser.parse_args()

#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
def find_project_root(start: Path = None, markers=(".git",)) -> Path:
    start = start or Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")


BASE_DIR = find_project_root()

RETRIEVAL_DIR = Path(f"{BASE_DIR}/localdata/test-A-02-retrievals") if args.test else Path(f"{BASE_DIR}/localdata/A-02-retrievals")
RETRIEVAL_LIST = list(RETRIEVAL_DIR.glob("*.pdf"))

OUTPUT_DIR    = Path(f"{BASE_DIR}/PipelineA-Answers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE  = OUTPUT_DIR / "results.json"

DPI = 150

# access models=[
    # 'Qwen3.5-35B-A3B'
    # 'gemma-4-31B-it'
    # 'gemma-3-27b-it'
    # 'gpt-oss-120b'
    # ]

MODEL_NAME = "gemma-4-31B-it"

PROMT_PATH = Path(f"{BASE_DIR}/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
EXTRACTION_PROMT = PROMT_PATH.read_text()



print(f"RETRIEVAL_DIR:  {RETRIEVAL_DIR}")
print(f"OUTPUT_DIR:     {OUTPUT_DIR}")
print(f"PROMT_PATH:     {PROMT_PATH}")
print(f"EXTRACTION_PROMT:\n{EXTRACTION_PROMT}\n")
print("=" * 60)
print()


#### 1. PROCESSING PDFS ########################################
banner("STEP 1: PROCESSING PDFS")

client = OpenAI(
    base_url="https://gpt.uni-muenster.de/v1"
)

results = []
n = len(RETRIEVAL_LIST)
counter = 1

for pdf_path in sorted(RETRIEVAL_LIST):
    
    report_name = pdf_path.stem
    print(report_name)
    
    current_pdf_imgages = []
    with fitz.open(str(pdf_path)) as doc :
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDF is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            current_pdf_imgages.append(img)


    print("    PDF2Image done.")
    
    content = [{
        "type": "text",
        "text": EXTRACTION_PROMT
    }]
    for img in current_pdf_imgages:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    
    print("    Submitting API Request")
    comletion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": content
        }],
        text={"format": {"type": "json_object"}}
    )

    print(comletion.choices[0].message.content)
    
    result = {
        "report"    : report_name,
        "response"  : comletion.output_text,
        "model"     : MODEL_NAME,
        "usage": {
            "input_tokens": comletion.usage.input_tokens,
            "output_tokens": comletion.usage.output_tokens,
        }
    }
    results.append(result)
    
    print(f"{pdf_path} processed. {counter} / {n}")
    counter = counter + 1

banner("Saving results")

RESULTS_FILE.write_text(json.dumps(results, indent=2))
print(f"  Results saved to: {RESULTS_FILE}")
print(f"  Total requests: {len(results)}")