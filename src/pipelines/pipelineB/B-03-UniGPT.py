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

# access models=[
    # 'Qwen3.5-35B-A3B'
    # 'gemma-4-31B-it'
    # 'gemma-3-27b-it'
    # 'gpt-oss-120b' 'TEXT ONLY'
    # ]
MODEL_NAME = "gemma-4-31B-it" # fastest
#MODEL_NAME = "Qwen3.5-35B-A3B" # slowest, but thinking
#MODEL_NAME = "gemma-3-27b-it" # longer than 4, faster than Qwen


BASE_DIR = find_project_root()

RETRIEVAL_DIR = Path(f"{BASE_DIR}/localdata/test-A-02-retrievals") if args.test else Path(f"{BASE_DIR}/localdata/A-02-retrievals")
RETRIEVAL_LIST = list(RETRIEVAL_DIR.glob("*.pdf"))

OUTPUT_DIR    = Path(f"{BASE_DIR}/src/pipelines/pipelineB/PipelineB-Answers/{MODEL_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE  = OUTPUT_DIR / "results.json"

DPI = 150

PROMT_PATH = Path(f"{BASE_DIR}/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
EXTRACTION_PROMT = PROMT_PATH.read_text()



print(f"RETRIEVAL_DIR:  {RETRIEVAL_DIR}")
print(f"OUTPUT_DIR:     {OUTPUT_DIR}")
print(f"PROMT_PATH:     {PROMT_PATH}")
print(f"MODEL_NAME:     {MODEL_NAME}")
# print(f"EXTRACTION_PROMT:\n{EXTRACTION_PROMT}\n")
print()


#### 1. PROCESSING PDFS ########################################
banner("PROCESSING PDFS")

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
    
    print("    Base64 encoding done.")
    
    
    
    
    print("    Submitting API Request")
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"}
    )
    
    raw_completion = completion.choices[0].message.content
    
    clean_completion = raw_completion.strip()
    if clean_completion.startswith("```json"):
        clean_completion = clean_completion[7:-3].strip()
    elif clean_completion.startswith("```"):
        clean_completion = clean_completion[3:-3].strip()



    output = json.loads(clean_completion)

    output_file = OUTPUT_DIR / f"{report_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)



    print(f"    Saved to {output_file}")
    print()
    print(f"{pdf_path} processed. {counter} / {n}")
    counter = counter + 1

banner("Done.")

#banner("Saving results")

#RESULTS_FILE.write_text(json.dumps(results, indent=2))
#print(f"  Results saved to: {RESULTS_FILE}")
#print(f"  Total requests: {len(results)}")