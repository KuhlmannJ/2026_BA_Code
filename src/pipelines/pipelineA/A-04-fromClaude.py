import argparse
import json
import os
import time

from pathlib import Path
import anthropic

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
parser.add_argument("--condition", "-c", choices=["bare", "thinking", "thinking_system"], default="thinking_system", help="Run condition: bare | thinking | thinking_system (must match A-03)")
args = parser.parse_args()

#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

#### 0. GLOBAL VARIABLES ########################################

BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located

# Condition selects the same per-condition folder A-03 wrote to
CONDITION = args.condition

OUTPUT_DIR    = Path(BASE) / "../../../evaluations/PipelineA/PipelineA-Answers" / CONDITION
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

POLL_INTERVAL = 1  # seconds between status checks


batch_id = BATCH_ID_FILE.read_text().strip()
client   = anthropic.Anthropic()

(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

banner(f"Polling batch [{CONDITION}]: {batch_id}")

# Poll until done
while True:
    batch  = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(
        f"  Status: {batch.processing_status} — "
        f"succeeded: {counts.succeeded}, "
        f"errored: {counts.errored}, "
        f"processing: {counts.processing}"
    )

    if batch.processing_status == "ended":
        break

    print(f"  Retrying in {POLL_INTERVAL}s...")
    time.sleep(POLL_INTERVAL)

banner("Retrieving results")

succeeded, errored = 0, 0

total_input, total_output, total_cost = 0, 0, 0

for entry in client.messages.batches.results(batch_id):
    report_name = entry.custom_id

    if entry.result.type != "succeeded":
        print(f"[ERR]  {report_name}: {entry.result.type}")
        errored += 1
        continue
    
    # Finding the actual text response in the whole asnwer, because there are
    # also other blocks like 'thinking'.
    text_blocks = [b.text for b in entry.result.message.content if b.type == "text"]
    if not text_blocks:
        print(f"[ERR]  {report_name}: no text block in response")
        errored += 1
        continue
    raw = text_blocks[0]

    # Step 4 — Strip down the JSON format for puring into a file
    clean = raw.strip()
    if clean.startswith("```json"):
        clean = clean[7:-3].strip()
    elif clean.startswith("```"):
        clean = clean[3:-3].strip()

    output_path = OUTPUT_DIR / f"{report_name}.json"
    output_path.write_text(
        json.dumps(json.loads(clean), indent=2, ensure_ascii=False)
    )
    
    
    #### Some calulations how expensive the report was
    input_tokens  = entry.result.message.usage.input_tokens
    output_tokens = entry.result.message.usage.output_tokens
    
    total_input  += input_tokens
    total_output += output_tokens
    
    input_cost    = (input_tokens  / 1_000_000) * 2.97
    output_cost   = (output_tokens / 1_000_000) * 14.88
    
    total_cost   += input_cost + output_cost
    
    
    #### Puring all into a log for maybe later use
    
    log_path = OUTPUT_DIR / f"logs/{report_name}.log"
    log_path.write_text(json.dumps({
        "report":       report_name,
        "condition":    CONDITION,
        "batch_id":     batch_id,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "total_cost_USD": input_cost + output_cost,
        "raw_response": raw,
    }, indent=2, ensure_ascii=False))

    succeeded += 1
    print(f"[OK]   {report_name}")

banner(f"Done [{CONDITION}]: {succeeded} succeeded, {errored} errored")
banner(f"Input: {total_input} | Output: {total_output} | Cost: {round(total_cost,2)}")
print(f"  Results in: {OUTPUT_DIR}")