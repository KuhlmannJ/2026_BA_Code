import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

import argparse
import json
import time
import csv

import fitz #pip install pymupdf
from PIL import Image
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"   # VRAM-ERROR, 500GB download :)
# MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"         # 66.7GB VRAM, takes 5min/report, 2nd 88GB VRAM, 67.21 GB
# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"     # 62.1GB VRAM
# MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"         # NON-Thinking
# MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct-FP8"     # NON-Thinking-FP8


# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test",       "-t",   action="store_true" , help="Toggle Testing Path")
parser.add_argument("--maxTokens",  "-mt",  type=int, default=16384, help="Control Thinking Tokens")
# A hard-limit on the length of the thought process
# 16384 was seen in some HF examples of the authors, besides 128 (way too small)

parser.add_argument("--gepaTrainSet", "-gt", action="store_true", help="Toggle Training Set of Reports")

parser.add_argument("--model", "-m",
                    choices=["think", "moe", "instr", "instrFP8", "intr8B"],
                    default="think",
                    help="Model to use: %(choices)s")

parser.add_argument("--prompt-file", "-p",
                    type=Path, default=None,
                    help="Path to prompt .txt file (overrides default BaselineA-Prompt.txt)")

parser.add_argument("--output-dir", "-o",
                    type=Path, default=None,
                    help="Output directory for JSON results (overrides default scratch path)")

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

banner("START: B-03-HPC.py")

#### 0. GLOBAL VARIABLES ########################################
banner("STEP 0: GLOBAL VARIABLES")

MAX_TOKENS = args.maxTokens
match args.model:
    case "think":       MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"
    case "moe":         MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    case "instr":       MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
    case "instrFP8":    MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    case "intr8B":      MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


# NOTE: Fixed RETRIEVAL_DIR for all models!
match True:
    case args.gepaTrainSet:
        RETRIEVAL_DIR = Path("/scratch/tmp/jkuhlma1/gepa/gepaTrainSet/")
    case args.test:
        RETRIEVAL_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/test/nvidia/nemotron-colembed-vl-8b-v2/")
    case _:
        RETRIEVAL_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/")

RETRIEVAL_LIST = sorted(list(RETRIEVAL_DIR.glob("*.pdf")))


if args.output_dir is not None:
    OUTPUT_DIR = args.output_dir
elif args.test:
    OUTPUT_DIR = Path(f"/scratch/tmp/jkuhlma1/results/B-03-answers/test/{MODEL_NAME}")
else:
    OUTPUT_DIR = Path(f"/scratch/tmp/jkuhlma1/results/B-03-answers/{MODEL_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE  = OUTPUT_DIR / "results.json"


DPI = 150
TIME_ROUND = 2 # Rounding for time logging

PROMT_PATH = (
    args.prompt_file
    if args.prompt_file is not None
    else Path("/home/j/jkuhlma1/2026_BA_Code/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
)
EXTRACTION_PROMT = PROMT_PATH.read_text()

print(f"RETRIEVAL_DIR:  {RETRIEVAL_DIR}")
print(f"No. of PDF:     {len(RETRIEVAL_LIST)}")
print(f"OUTPUT_DIR:     {OUTPUT_DIR}")
print(f"PROMT_PATH:     {PROMT_PATH}")
print(f"MODEL_NAME:     {MODEL_NAME}")
print(f"Max Tokens:     {args.maxTokens}")
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

num_gpus = torch.cuda.device_count()
print(f"  Available GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)")



#### 2. VLM Loading #############################################
banner("STEP 2: LOAD VLM")

t_vlmLoad_start = time.time()

match args.model:
    case "think":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    case "moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    case "instr":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    case "instrFP8":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float8_e4m3fn,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    case "intr8B":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

print("\n=== Model Device Mapping ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
    break  # nur erste Parameter prüfen

processor = AutoProcessor.from_pretrained(MODEL_NAME)

t_vlmLoad = round(time.time() - t_vlmLoad_start, TIME_ROUND)

print(f" Loaded: {MODEL_NAME} in {t_vlmLoad} s")
print(f" Attention loaded:{model.config._attn_implementation}")
print(f" VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")




#### 3. PDF PROCESSING ########################################
banner("STEP 3: PDF PROCESSING")

n = len(RETRIEVAL_LIST)
counter = 1
results = []

for pdf_path in sorted(RETRIEVAL_LIST):
    t_pdf_start = time.time()
    
    report_name = pdf_path.stem
    print(report_name)
    report_len = 0

    t_pymupdf_start = time.time()
    #### Packaging Promt and Images (as img as no API call))
    content = []
    with fitz.open(str(pdf_path)) as doc :
        report_len = len(doc)
        for page in doc :
            pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDF is RGBA (transparent)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": EXTRACTION_PROMT}) # Promt after img, like on HF
    
    messages = [{"role": "user", "content": content}]       
    print("    PDF2Image done and embedded into `content` and `messages`.")
    t_pymupdf = round(time.time() - t_pymupdf_start, TIME_ROUND)
    print(f"    t_pymupdf: {t_pymupdf}s")
    
    # Preparation for inference (source: HF)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda") #"cuda" better for multi H200mini GPUs than (model.device)
    
    output_JSON = None  # Default: to detect token-overflow
    tokens_needed = MAX_TOKENS
    # This loop allows for one(!) retry of VLM extraction with double the tokens if needed
    for attempt, now_max_tokens in enumerate([MAX_TOKENS, MAX_TOKENS * 2]):
        print(f"    Attempt: {attempt} with {now_max_tokens} Token-Limit")
        t_inference_start = time.time()
        # Inference: Generation of the output (source: HF)
        generated_ids = model.generate(**inputs, max_new_tokens=now_max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        t_inference = round(time.time() - t_inference_start, TIME_ROUND)
        print(f"    t_inference: {t_inference}s")
        
        
        t_processorbatch_start = time.time()
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False #skip_special_tokens=FALSE um <think> zu lassen
        )[0] # To get to the String inside the output_text: >>["So, let's describe..."]<<
        t_processorbatch = round(time.time() - t_processorbatch_start, TIME_ROUND)
        print(f"    t_processorbatch: {t_processorbatch}s")
        
        # Cleanup of output text 
        # strip_thinking() drops "thinking" part of the response
        # An "<|im_end|>" is always at the end of the output, needs to be removed
        # .strip() drops random empty lines
        output_clean = strip_thinking(output_text).replace("<|im_end|>", "").strip()
        
        try:
            output_JSON = json.loads(output_clean)
            break
        except json.JSONDecodeError:
            if attempt == 0: #attempt += 1 with for-loop
                print(f"  [WARN] JSON failed, retrying with {now_max_tokens * 2} tokens...")
                tokens_needed = MAX_TOKENS * 2
            else:
                print(f"  [ERROR] JSON failed after retry, skipping {report_name}") #Leaves for loop with output_JSON = None
        
    # If token-overflow => Skip through next 
    if output_JSON is not None:
    
        # with open(f"{report_name}_outout_without_thinking.txt", "w", encoding="utf-8") as f:
        #     f.write(output_clean)
        output_JSON = json.loads(output_clean)

        # Saving that outout as JSON
        output_file = OUTPUT_DIR / f"{report_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_JSON, f, ensure_ascii=False, indent=2)
    
    
        results.append({
            "model":        MODEL_NAME,
            "maxToken":     args.maxTokens,
            "report":       report_name,
            "duration":     t_inference,
            "pages":        report_len,
            "t_inf/page":   round(t_inference / report_len, TIME_ROUND),
            
        })
            
        t_pdf = round(time.time() - t_pdf_start, TIME_ROUND)
        print(f"    t_pdf: {t_pdf}s")
        print(f"    Saved to {output_file}")
        print( "    Report processed.")
        print(f"    {counter} / {n}")
        print()
        counter += 1


fieldnames = ["model", "maxToken", "report", "duration", "pages", "t_inf/page"]
csv_file = OUTPUT_DIR / "***results.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

banner("END: B-03-HPC.py")