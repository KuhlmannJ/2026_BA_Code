import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

import argparse
import json
import re
import time

import fitz #pip install pymupdf
from PIL import Image
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test",  "-t", action="store_true",       help="Toggle Testing Path")
args = parser.parse_args()




#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
def clean_json(text: str) -> str:
    text = text.replace("<|im_end|>", "").strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()
    return text

# Drops everything before </think>
def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

# MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"   # VRAM-ERROR, 500GB download :)
MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"           # 66.7GB VRAM
# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"     # 62.1GB VRAM

# NOTE: Fixed RETRIEVAL_DIR!
RETRIEVAL_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/")
RETRIEVAL_LIST = sorted(list(RETRIEVAL_DIR.glob("*.pdf")))

with open("RETRIEVAL_LIST.txt", "w", encoding="utf-8") as f:
        f.write(RETRIEVAL_LIST)
# For Testing just one, hopefully 'Allianz_2022_report.pdf'
if args.test:
    RETRIEVAL_LIST = [RETRIEVAL_LIST[4]]

OUTPUT_DIR    = Path(f"/scratch/tmp/jkuhlma1/results/A-03-answers/{MODEL_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE  = OUTPUT_DIR / "results.json"

DPI = 150
TIME_ROUND = 2 # Rounding for time logging

PROMT_PATH = Path("/home/j/jkuhlma1/2026_BA_Code/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")

EXTRACTION_PROMT = PROMT_PATH.read_text()

print(f"RETRIEVAL_DIR:  {RETRIEVAL_DIR}")
print(f"No. of PDF:     {len(RETRIEVAL_LIST)}")
print(f"OUTPUT_DIR:     {OUTPUT_DIR}")
print(f"PROMT_PATH:     {PROMT_PATH}")
print(f"MODEL_NAME:     {MODEL_NAME}")
# print(f"EXTRACTION_PROMT:\n{EXTRACTION_PROMT}\n")
print()

if args.test :
    banner("THIS IS A TEST-RUN")
#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
props      = torch.cuda.get_device_properties(0)
gpu_name   = torch.cuda.get_device_name(0)
vram_total = props.total_memory / 1e9
gpu_uuid   = props.uuid
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  UUID : {gpu_uuid}")




#### 2. VLM Loading #############################################
banner("STEP 2: LOAD VLM")

match MODEL_NAME:
    case "Qwen/Qwen3-VL-32B-Thinking":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0',
        )
    case "Qwen/Qwen3-VL-30B-A3B-Thinking":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0',
        )

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f" Loaded: {MODEL_NAME}")
print(f" Attention loaded:{model.config._attn_implementation}")
print(f" VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")




#### 3. PDF PROCESSING ########################################
banner("STEP 3: PDF PROCESSING")

n = len(RETRIEVAL_LIST)
counter = 1

for pdf_path in sorted(RETRIEVAL_LIST):
    t_pdf_start = time.time()
    
    report_name = pdf_path.stem
    print(report_name)

    t_pymupdf_start = time.time()
    #### Packaging Promt and Images (as img as no API call))
    content = []
    with fitz.open(str(pdf_path)) as doc :
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDF is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": EXTRACTION_PROMT}) # Promt after img, like on HF
    
    messages = [{"role": "user", "content": content}]       
    print("    PDF2Image done and embedded into `content` and `messages`.")
    t_pymupdf = round(time.time() - t_pymupdf_start, TIME_ROUND)
    print(f"t_pymupdf: {t_pymupdf}s")
    
    # Preparation for inference (source: HF)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    
    t_inference_start = time.time()
    # Inference: Generation of the output (source: HF)
    generated_ids = model.generate(**inputs, max_new_tokens=40960)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    t_inference = round(time.time() - t_inference_start, TIME_ROUND)
    print(f"t_inference: {t_inference}s")
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False #skip_special_tokens=FALSE um <think> zu lassen
    )[0] # To get to the String inside the output_text: >>["So, let's describe..."]<<
    
    with open(f"{report_name}_raw.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    
    # Cleanup of output text that should be JSON #.strip() drops random empty lines
    output_clean = strip_thinking(output_text).strip()
    with open(f"{report_name}_outout_without_thinking.txt", "w", encoding="utf-8") as f:
        f.write(output_clean)
        
    output_JSON = json.loads(output_clean)

    # Saving that outout as JSON
    output_file = OUTPUT_DIR / f"{report_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_JSON, f, ensure_ascii=False, indent=2)
        
        
    t_pdf = round(time.time() - t_pdf_start, TIME_ROUND)
    print(f"t_pdf: {t_pdf}s")
    print(f"    Saved to {output_file}")
    print()
    print(f"{pdf_path} processed. {counter} / {n}")
    counter += 1

banner("Done.")