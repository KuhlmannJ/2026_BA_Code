import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

import json
import time
import csv

import fitz
from PIL import Image
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

TIME_ROUND = 2


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


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def load_model(model_arg: str) -> tuple:
    """Load model and processor once; return (model, processor, model_name)."""
    match model_arg:
        case "think":    model_name = "Qwen/Qwen3-VL-32B-Thinking"
        case "moe":      model_name = "Qwen/Qwen3-VL-30B-A3B-Thinking"
        case "instr":    model_name = "Qwen/Qwen3-VL-32B-Instruct"
        case "instrFP8": model_name = "Qwen/Qwen3-VL-32B-Instruct-FP8"
        case "instr8B":  model_name = "Qwen/Qwen3-VL-8B-Instruct"
        case _:          raise ValueError(f"Unknown model: {model_arg}")

    banner(f"LOAD VLM: {model_name}")
    t_start = time.time()

    if model_arg == "moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    elif model_arg == "instrFP8":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_name)
    t_load = round(time.time() - t_start, TIME_ROUND)

    print(f" Loaded: {model_name} in {t_load}s")
    print(f" Attention: {model.config._attn_implementation}")
    print(f" VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    return model, processor, model_name


def run_extraction(
    model,
    processor,
    model_name: str,
    extraction_prompt: str,
    output_dir: Path,
    retrieval_dir: Path,
    max_tokens: int = 32768, #Doubled, as it was only a ceeling getting hit
    dpi: int = 150,
) -> None:
    """Run VLM extraction on all PDFs in retrieval_dir; write per-report JSON and results.csv to output_dir."""
    output_dir = Path(output_dir)
    retrieval_dir = Path(retrieval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_list = sorted(retrieval_dir.glob("*.pdf"))
    n = len(retrieval_list)
    results = []
    counter = 1

    banner("PDF PROCESSING")
    for pdf_path in retrieval_list:
        t_pdf_start = time.time()
        report_name = pdf_path.stem
        print(report_name)
        report_len = 0

        t_pymupdf_start = time.time()
        content = []
        with fitz.open(str(pdf_path)) as doc:
            report_len = len(doc)
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": extraction_prompt})

        messages = [{"role": "user", "content": content}]
        t_pymupdf = round(time.time() - t_pymupdf_start, TIME_ROUND)
        print(f"    PDF2Image done. t_pymupdf: {t_pymupdf}s")

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        output_JSON = None
        t_inference = 0

        for attempt, now_max_tokens in enumerate([max_tokens, max_tokens * 2]):
            print(f"    Attempt: {attempt} with {now_max_tokens} Token-Limit")
            t_inference_start = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=now_max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            t_inference = round(time.time() - t_inference_start, TIME_ROUND)
            print(f"    t_inference: {t_inference}s")

            t_processorbatch_start = time.time()
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            t_processorbatch = round(time.time() - t_processorbatch_start, TIME_ROUND)
            print(f"    t_processorbatch: {t_processorbatch}s")

            output_clean = clean_json(strip_thinking(output_text))

            try:
                output_JSON = json.loads(output_clean)
                break
            except json.JSONDecodeError:
                if attempt == 0:
                    print(f"  [WARN] JSON failed, retrying with {now_max_tokens * 2} tokens...")
                    banner("JSON ERROR START")
                    print(output_clean)
                    banner("JSON ERROR END")
                else:
                    print(f"  [ERROR] JSON failed after retry, skipping {report_name}")

        if output_JSON is not None:
            output_file = output_dir / f"{report_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_JSON, f, ensure_ascii=False, indent=2)

            results.append({
                "model":      model_name,
                "maxToken":   max_tokens,
                "report":     report_name,
                "duration":   t_inference,
                "pages":      report_len,
                "t_inf/page": round(t_inference / report_len, TIME_ROUND),
            })

            t_pdf = round(time.time() - t_pdf_start, TIME_ROUND)
            print(f"    t_pdf: {t_pdf}s")
            print(f"    Saved to {output_file}")
            print(f"    {counter} / {n}")
            counter += 1

    fieldnames = ["model", "maxToken", "report", "duration", "pages", "t_inf/page"]
    csv_file = output_dir / "***results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test",         "-t",  action="store_true",     help="Toggle Testing Path")
    parser.add_argument("--maxTokens",    "-mt", type=int, default=16384, help="Control Thinking Tokens")
    parser.add_argument("--gepaTrainSet", "-gt", action="store_true",     help="Toggle Training Set of Reports")
    parser.add_argument("--model",        "-m",
                        choices=["think", "moe", "instr", "instrFP8", "instr8B"],
                        default="think",
                        help="Model to use: %(choices)s")
    parser.add_argument("--prompt-file",  "-p", type=Path, default=None,
                        help="Path to prompt .txt file (overrides default)")
    parser.add_argument("--output-dir",   "-o", type=Path, default=None,
                        help="Output directory for JSON results")
    args = parser.parse_args()

    banner("START: B_03_HPC_fn.py")
    banner("STEP 0: GLOBAL VARIABLES")

    if args.gepaTrainSet:
        retrieval_dir = Path("/scratch/tmp/jkuhlma1/gepa/gepaTrainSet/")
    elif args.test:
        retrieval_dir = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/test/nvidia/nemotron-colembed-vl-8b-v2/")
    else:
        retrieval_dir = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals/nvidia/nemotron-colembed-vl-8b-v2/")

    prompt_path = (
        args.prompt_file
        if args.prompt_file is not None
        else Path("/home/j/jkuhlma1/2026_BA_Code/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
    )
    extraction_prompt = prompt_path.read_text()

    print(f"RETRIEVAL_DIR:  {retrieval_dir}")
    print(f"No. of PDF:     {len(sorted(retrieval_dir.glob('*.pdf')))}")
    print(f"PROMPT_PATH:    {prompt_path}")
    print(f"Max Tokens:     {args.maxTokens}")

    if args.test:
        banner("THIS IS A TEST-RUN")

    banner("STEP 1: GPU / CUDA")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
    print(f"  UUID : {props.uuid}")
    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({p.total_memory / 1e9:.1f} GB)")

    model, processor, model_name = load_model(args.model)

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.test:
        output_dir = Path(f"/scratch/tmp/jkuhlma1/results/B-03-answers/test/{model_name}")
    else:
        output_dir = Path(f"/scratch/tmp/jkuhlma1/results/B-03-answers/{model_name}")

    print(f"OUTPUT_DIR:     {output_dir}")

    run_extraction(
        model=model,
        processor=processor,
        model_name=model_name,
        extraction_prompt=extraction_prompt,
        output_dir=output_dir,
        retrieval_dir=retrieval_dir,
        max_tokens=args.maxTokens,
    )

    banner("END: B_03_HPC_fn.py")
