import argparse
import json
import time

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

OUTPUT_DIR    = Path("PipelineA-Answers")
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

POLL_INTERVAL = 1  # seconds between status checks

#### Main ########################################################

def main() -> None:

    if not BATCH_ID_FILE.exists():
        print(f"[ERROR] {BATCH_ID_FILE} not found. Run A-03-toClaude.py first.")
        return

    batch_id = BATCH_ID_FILE.read_text().strip()
    client   = anthropic.Anthropic()

    banner(f"Polling batch: {batch_id}")

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

        raw = entry.result.message.content[0].text

        # Step 4 — Persist (same as A-02)
        clean = raw.strip()
        if clean.startswith("```json"):
            clean = clean[7:-3].strip()
        elif clean.startswith("```"):
            clean = clean[3:-3].strip()

        output_path = OUTPUT_DIR / f"{report_name}.json"
        output_path.write_text(
            json.dumps(json.loads(clean), indent=2, ensure_ascii=False)
        )
        
        input_tokens  = entry.result.message.usage.input_tokens
        output_tokens = entry.result.message.usage.output_tokens
        
        total_input  += input_tokens
        total_output += output_tokens
        
        input_cost    = (input_tokens  / 1_000_000) * 2.97
        output_cost   = (output_tokens / 1_000_000) * 14.88
        
        total_cost   += input_cost + output_cost
        
        log_path = OUTPUT_DIR / f"logs/{report_name}.log"
        log_path.write_text(json.dumps({
            "report":       report_name,
            "batch_id":     batch_id,
            "input_tokens":   input_tokens,
            "output_tokens":  output_tokens,
            "total_cost_USD": input_cost + output_cost,
            "raw_response": raw,
        }, indent=2, ensure_ascii=False))

        succeeded += 1
        print(f"[OK]   {report_name}")

    banner(f"Done: {succeeded} succeeded, {errored} errored")
    banner(f"Input: {total_input} | Output: {total_output} | Cost: {round(total_cost,2)}")
    print(f"  Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()