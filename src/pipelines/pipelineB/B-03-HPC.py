#########
## Da nur 30% Auslastung der H200 bei 80GB VRAM Belegung, Batch-Processing in Erwägung ziehen

# for chunk in pdf_chunks:
#     texts, all_images, names = [], [], []

#     for pdf_path in chunk:
#         images = load_pdf_images(pdf_path)
#         content = [{"type": "image", "image": img} for img in images]
#         content.append({"type": "text", "text": EXTRACTION_PROMPT})
#         messages = [{"role": "user", "content": content}]

#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         texts.append(text)
#         all_images.extend(images)
#         names.append(pdf_path.stem)

#     inputs = processor(
#         text=texts,
#         images=all_images,
#         padding=True,
#         return_tensors="pt"
#     ).to(model.device)

#     with torch.no_grad(): ### PyTorch baut während model.generate() keinen Computation Graph, bei BATCH_SIZE > 1 wohl bemerkbar
#         generated_ids = model.generate(**inputs, max_new_tokens=16384)

#     for i, (gen_ids, inp_ids, name) in enumerate(zip(generated_ids, inputs.input_ids, names)):
#         trimmed = gen_ids[len(inp_ids):]
#         raw = processor.decode(trimmed, skip_special_tokens=False)
#         clean = strip_thinking(raw).replace("<|im_end|>", "").strip()
#         # json.loads + save ...




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


# ── Arguments for Dev'ing
parser = argparse.ArgumentParser()
parser.add_argument("--test",       "-t",   action="store_true", help="Toggle Testing Path")
parser.add_argument("--batch_size", "-bz",  type=int, default=2)
parser.add_argument("--maxTokens",  "-mt",  type=int, default=16384, help="Control Thinking Tokens")
#Rougly equivalent to control thinking tokens, but means tokens overall
# 16384 was seen in some HF examples of the authors, besides 128 (way too small)

args = parser.parse_args()




#### Helping Functions ##########################################
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
def load_pdf_images(pdf_path):
    images = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=DPI, alpha=False) # If PDF is RGBA (transparent)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images
    
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
# MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"           # 66.7GB VRAM, takes 5min/report
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"     # 62.1GB VRAM, took 03:58:12 for 53 reports

BATCH_SIZE = args.batch_size

# NOTE: Fixed RETRIEVAL_DIR!
RETRIEVAL_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/")
RETRIEVAL_LIST = sorted(list(RETRIEVAL_DIR.glob("*.pdf")))

# For Testing just one
if args.test:
    RETRIEVAL_LIST = [RETRIEVAL_LIST[0]]

OUTPUT_DIR    = Path(f"/scratch/tmp/jkuhlma1/results/A-03-answers/{MODEL_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE  = OUTPUT_DIR / "results.json"

DPI = 150
TIME_ROUND = 2 # Rounding for time logging

PROMT_PATH = Path("/home/j/jkuhlma1/2026_BA_Code/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")

EXTRACTION_PROMPT = PROMT_PATH.read_text()

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

t_vlmLoad_start = time.time()
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

t_vlmLoad = round(time.time() - t_vlmLoad_start, TIME_ROUND)

print(f" Loaded: {MODEL_NAME} in {t_vlmLoad} s")
print(f" Attention loaded:{model.config._attn_implementation}")
print(f" VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")




#### 3. PDF PROCESSING ########################################
banner("STEP 3: PDF PROCESSING")

n = len(RETRIEVAL_LIST)
counter = 1
results = []

# Batch-Loop
pdf_chunks = [RETRIEVAL_LIST[i:i+BATCH_SIZE] for i in range(0, len(RETRIEVAL_LIST), BATCH_SIZE)]

for chunk in pdf_chunks:
    texts, all_images, names = [], [], []

    for pdf_path in chunk:
        t_pymupdf_start = time.time()
        report_name = pdf_path.stem
        
        images = load_pdf_images(pdf_path)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": EXTRACTION_PROMPT})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        all_images.extend(images)
        names.append(report_name)
        
        t_pymupdf = round(time.time() - t_pymupdf_start, TIME_ROUND)
        print(f"t_pymupdf: {t_pymupdf}s")

    inputs = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    t_inference_start = time.time()
    with torch.no_grad(): ### PyTorch baut während model.generate() keinen Computation Graph, bei BATCH_SIZE > 1 wohl bemerkbar
        generated_ids = model.generate(**inputs, max_new_tokens=16384)
    
    
    for i, (gen_ids, inp_ids, name) in enumerate(zip(generated_ids, inputs.input_ids, names)):
        print(name)
        t_inference_start = time.time()
        trimmed = gen_ids[len(inp_ids):]
        
        output_text = processor.decode(trimmed, skip_special_tokens=False)
        t_inference = round(time.time() - t_inference_start, TIME_ROUND)
        print(f"    t_inference: {t_inference}s")
        
        
        output_clean = strip_thinking(output_text).replace("<|im_end|>", "").strip()
        output_JSON = json.loads(output_clean)

        # Saving that outout as JSON
        output_file = OUTPUT_DIR / f"{name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_JSON, f, ensure_ascii=False, indent=2)
        
        
        results.append({
            "model":        MODEL_NAME,
            "maxToken":     args.maxTokens,
            "report":       name,
            "duration":     t_inference,
            #"pages":        report_len,
            #"t_inf/page":   round(t_inference / report_len, TIME_ROUND),
            
        })
        
        print(f"    Saved to {output_file}")
        print()
        print(f"    {pdf_path} processed. {counter} / {n}")
        counter += 1


# for pdf_path in sorted(RETRIEVAL_LIST):
#     t_pdf_start = time.time()
    
#     report_name = pdf_path.stem
#     print(report_name)
#     report_len = 0

#     t_pymupdf_start = time.time()
#     #### Packaging Promt and Images (as img as no API call))
#     content = []
#     with fitz.open(str(pdf_path)) as doc :
#         report_len = len(doc)
#         for page in doc :
#             pix = page.get_pixmap(dpi = DPI, alpha=False) # If PDF is RGBA (transparent)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             content.append({"type": "image", "image": img})
#     content.append({"type": "text", "text": EXTRACTION_PROMPT}) # Promt after img, like on HF
    
#     messages = [{"role": "user", "content": content}]       
#     print("    PDF2Image done and embedded into `content` and `messages`.")
#     t_pymupdf = round(time.time() - t_pymupdf_start, TIME_ROUND)
#     print(f"t_pymupdf: {t_pymupdf}s")
    
#     # Preparation for inference (source: HF)
#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt"
#     ).to(model.device)
    
    
#     t_inference_start = time.time()
#     # Inference: Generation of the output (source: HF)
#     generated_ids = model.generate(**inputs, max_new_tokens=args.maxTokens)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     t_inference = round(time.time() - t_inference_start, TIME_ROUND)
#     print(f"t_inference: {t_inference}s")
    
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False #skip_special_tokens=FALSE um <think> zu lassen
#     )[0] # To get to the String inside the output_text: >>["So, let's describe..."]<<
    
#     # Cleanup of output text 
#     # strip_thinking() drops "thinking" part of the response
#     # An "<|im_end|>" is always at the end of the output, needs to be removed
#     # .strip() drops random empty lines
#     output_clean = strip_thinking(output_text).replace("<|im_end|>", "").strip()
        
#     output_JSON = json.loads(output_clean)

#     # Saving that outout as JSON
#     output_file = OUTPUT_DIR / f"{report_name}.json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(output_JSON, f, ensure_ascii=False, indent=2)
    
    
#     results.append({
#         "model":        MODEL_NAME,
#         "maxToken":     args.maxTokens,
#         "report":       report_name,
#         "duration":     t_inference,
#         "pages":        report_len,
#         "t_inf/page":   round(t_inference / report_len, TIME_ROUND),
        
#     })
        
#     t_pdf = round(time.time() - t_pdf_start, TIME_ROUND)
#     print(f"t_pdf: {t_pdf}s")
#     print(f"    Saved to {output_file}")
#     print()
#     print(f"{pdf_path} processed. {counter} / {n}")
#     counter += 1


fieldnames = ["model", "maxToken", "report", "duration", "pages", "t_inf/page"]
csv_file = OUTPUT_DIR / "***results.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

banner("Done.")