import argparse
import base64
import json
import os
import re

from pathlib import Path
import anthropic

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

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

RETRIEVAL_DIR = Path("../../../localdata/test-A-02-retrievals") if args.test else Path("../../../localdata/A-02-retrievals")
RETRIEVAL_LIST = list(RETRIEVAL_DIR.glob("*.pdf"))

OUTPUT_DIR    = Path("PipelineA-Answers")
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

MODEL_ID   = "claude-opus-4-7"
MAX_TOKENS = 8000

PROMT_PATH = Path("../../../baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
print(PROMT_PATH)
EXTRACTION_PROMT = PROMT_PATH.read_text()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#### 1. BUILDING BATCH ########################################
banner("STEP 1: BUILDING BATCH")

client = anthropic.Anthropic()

# Skip reports that already have results
already_done = {p.stem for p in OUTPUT_DIR.glob("*.json")}

requests = []
n = len(RETRIEVAL_LIST)
counter = 1

for pdf_path in sorted(RETRIEVAL_LIST):
    if pdf_path.stem in already_done:
        print(f"[SKIP] {pdf_path.stem}")
        continue

    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()
    requests.append({
        "custom_id" : re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem)[:64],
        "params": {
            "model":      MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type":       "base64",
                            "media_type": "application/pdf",
                            "data":       pdf_b64,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMT},
                ],
            }],
        },
    })
    
    print(f"{pdf_path} queued. {counter} / {n}")
    counter = counter + 1

banner(f"Submitting {len(requests)} requests")

batch = client.messages.batches.create(requests=requests)

BATCH_ID_FILE.write_text(batch.id)
print(f"  Batch ID:  {batch.id}")
print(f"  Status:    {batch.processing_status}")
print(f"  Saved to:  {BATCH_ID_FILE}")