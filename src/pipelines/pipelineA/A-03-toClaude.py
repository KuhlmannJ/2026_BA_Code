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
parser.add_argument("--condition", "-c", choices=["bare", "thinking", "thinking_system"], default="thinking_system", help="Run condition: bare | thinking | thinking_system")
args = parser.parse_args()

#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located

# Condition toggles (one code state per run, selected via -c)
CONDITION  = args.condition
THINKING   = CONDITION in ("thinking", "thinking_system")
USE_SYSTEM = CONDITION == "thinking_system"
print(f"  Condition: {CONDITION}  (thinking={THINKING}, system={USE_SYSTEM})")

RETRIEVAL_DIR = Path(BASE) / "../../../localdata/test-A-02-retrievals" if args.test else Path(BASE) / "../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2"
RETRIEVAL_LIST = list(RETRIEVAL_DIR.glob("*.pdf"))

# One output subfolder per condition, so skip-logic and batch_id do not mix across runs
OUTPUT_DIR    = Path(BASE) / "../../../evaluations/PipelineA/PipelineA-Answers" / CONDITION
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

MODEL_ID   = "claude-opus-4-7"
MAX_TOKENS = 32768

PROMT_PATH = Path(BASE) / "../../../baselines/baseline_frontier_model/Baseline-Prompt.txt"
print(PROMT_PATH)
EXTRACTION_PROMT = PROMT_PATH.read_text()

SYSTEM_PROMPT_PATH = Path(BASE) / "system-prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if USE_SYSTEM else None

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

    params = {
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
    }

    # Condition-specific fields
    if THINKING:
        params["thinking"] = {"type": "adaptive"} #"display": "omitted" is the default
    if USE_SYSTEM:
        params["system"] = SYSTEM_PROMPT

    requests.append({
        "custom_id" : re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem)[:64],
        "params": params,
    })

    print(f"{pdf_path} queued. {counter} / {n}")
    counter = counter + 1

banner(f"Submitting {len(requests)} requests [{CONDITION}]")

batch = client.messages.batches.create(requests=requests)

BATCH_ID_FILE.write_text(batch.id)
print(f"  Condition: {CONDITION}")
print(f"  Batch ID:  {batch.id}")
print(f"  Status:    {batch.processing_status}")
print(f"  Saved to:  {BATCH_ID_FILE}")

#######################
### OUTPUT
# Condition is now encoded in the output path:
#   .../evaluations/PipelineA/PipelineA-Answers/<condition>/
# batch_id.txt and result JSONs live inside that per-condition folder.