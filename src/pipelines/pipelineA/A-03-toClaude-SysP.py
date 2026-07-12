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

BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located

RETRIEVAL_DIR = Path(BASE) / "../../../localdata/test-A-02-retrievals" if args.test else Path(BASE) / "../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2"
RETRIEVAL_LIST = list(RETRIEVAL_DIR.glob("*.pdf"))

OUTPUT_DIR    = Path(BASE) / "../../../evaluations/PipelineA/PipelineA-Answers"
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

MODEL_ID   = "claude-opus-4-7"
MAX_TOKENS = 32768

PROMT_PATH = Path(BASE) / "../../../baselines/baseline_frontier_model/Baseline-Prompt.txt"
print(PROMT_PATH)
EXTRACTION_PROMT = PROMT_PATH.read_text()

SYSTEM_PROMPT_PATH = Path(BASE) / "system-prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

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
            "thinking": {"type": "adaptive"}, #"display": "omitted" is the default
            "system":     SYSTEM_PROMPT,
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

#######################
### OUTPUT
# ============================================================
#   STEP 0: GLOBAL VARIABLES
# ============================================================
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../baselines/baseline_frontier_model/Baseline-Prompt.txt

# ============================================================
#   STEP 1: BUILDING BATCH
# ============================================================
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/Allianz_2022_report.pdf queued. 1 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/Daimler_2020_report.pdf queued. 2 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/Fresenius SE_2019_report.pdf queued. 3 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/ViacomCBS_ESG Report_2020-2021_vFINAL.pdf queued. 4 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/acuity brands inc_2022_report.pdf queued. 5 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/addtech_2022_report.pdf queued. 6 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/aixtron_2020_report.pdf queued. 7 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/allfunds group_2021_report.pdf queued. 8 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/americold realty inc_2022_report.pdf queued. 9 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/atlas arteria_2019_report.pdf queued. 10 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/autoneum holding_2019_report.pdf queued. 11 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/bekaert (d) sa_2022_report.pdf queued. 12 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/blackline safety_2021_report.pdf queued. 13 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/cabot corp_2018_report.pdf queued. 14 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/cardinal energy ltd_2021_report.pdf queued. 15 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/comerica inc_2019_report.pdf queued. 16 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/cytokinetics inc_2022_report.pdf queued. 17 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/dampskibsselskabet norden_2019_report.pdf queued. 18 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/evolution mining ltd_2020_report.pdf queued. 19 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/fujifilm_2022_report.pdf queued. 20 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/georg fischer ag_2018_report.pdf queued. 21 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/graincorp_2019_report.pdf queued. 22 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/granite construction inc_2020_report.pdf queued. 23 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/h. lundbeck_2021_report.pdf queued. 24 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/hammerson reit_2021_report.pdf queued. 25 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/hang lung properties_2018_report.pdf queued. 26 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/healius ltd_2022_report.pdf queued. 27 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/hudbay minerals inc_2020_report.pdf queued. 28 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/huhtamaki_2018_report.pdf queued. 29 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/huhtamaki_2019_report.pdf queued. 30 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/inchcape plc_2022_report.pdf queued. 31 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/innospec inc_2020_report.pdf queued. 32 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/jetblue airways corp_2019_report.pdf queued. 33 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/jpmorgan chase_2020_report.pdf queued. 34 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/kfw_2018_report.pdf queued. 35 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/kilroy realty_2017_report.pdf queued. 36 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/kilroy realty_2018_report.pdf queued. 37 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/kureha corp_2020_report.pdf queued. 38 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/lundin gold inc_2021_report.pdf queued. 39 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/lundin gold inc_2022_report.pdf queued. 40 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/metro ag_2022_report.pdf queued. 41 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/morinaga ltd_2020_report.pdf queued. 42 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/nippn corp_2022_report.pdf queued. 43 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/nok corporation_2021_report.pdf queued. 44 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/nordea_2017_report.pdf queued. 45 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/polaris_2021_report.pdf queued. 46 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/salzgitter ag_2018_report.pdf queued. 47 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/sato oyj_2022_report.pdf queued. 48 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/stantec inc_2019_report.pdf queued. 49 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/sumitomo corporation_2021_report.pdf queued. 50 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/toshiba tec corp_2022_report.pdf queued. 51 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/uniper_2019_report.pdf queued. 52 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/varta ag_2021_report.pdf queued. 53 / 54
# /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../localdata/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/webuild_2019_report.pdf queued. 54 / 54

# ============================================================
#   Submitting 54 requests
# ============================================================
#   Batch ID:  msgbatch_017RTUsKE2CWakhmnzVk6s53
#   Status:    in_progress
#   Saved to:  /Users/jannikkuhlmann/VSC/LaTeX/2026_BA_Code/src/pipelines/pipelineA/../../../evaluations/PipelineA/PipelineA-Answers/batch_id.txt
#######################