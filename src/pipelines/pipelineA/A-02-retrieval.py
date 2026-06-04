import argparse
import base64
import json
import os
from pathlib import Path

import fitz  # PyMuPDF
import torch

# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--test", "-t", action="store_true", help="Toggle Testing Path")
parser.add_argument("--batch_size", "-bz", type=int, default=2)
args = parser.parse_args()

TOP_K      = 3
MODEL_ID   = "claude-opus-4-7"
MAX_TOKENS = 8000

PDF_DIR = Path("/scratch/tmp/jkuhlma1/data/test_esg_reports") if args.test else Path("/scratch/tmp/jkuhlma1/data/esg_reports")
EMB_DIR = Path("/scratch/tmp/jkuhlma1/data/embeddings/test_embeddings_colembed_3b_v2") if args.test else Path("/scratch/tmp/jkuhlma1/data/embeddings/embeddings_colembed_3b_v2")

OUTPUT_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-answers")
RESULT_DIR = Path("/scratch/tmp/jkuhlma1/results/A-02-retrievals")

# Retrieval query — placeholder for optimize_anything / GEPA optimization
# TODO: replace with optimized query once GEPA iterations are complete
RETRIEVAL_QUERY = (
    "What are the total CO2 equivalent greenhouse gas emissions? "
    "Include Scope 1, Scope 2 (market-based and location-based), "
    "and Scope 3 emissions with their values, units, and reporting years."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def select_pages(scores: torch.Tensor, top_k: int = TOP_K) -> list[int]:
    """Top-k pages by score, expanded with ±1 neighbors (Beck et al.)."""
    n = len(scores)
    top_idx = scores.topk(min(top_k, n)).indices.tolist()
    pages: set[int] = set()
    for idx in top_idx:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < n:
                pages.add(neighbor)
    return sorted(pages)


def extract_pages_as_pdf(pdf_path: Path, page_indices: list[int], save_path: Path) -> bytes:
    """Extract selected pages into a mini-PDF, save it, and return the bytes."""
    src = fitz.open(str(pdf_path))
    out = fitz.open()
    out.insert_pdf(src, from_page=min(page_indices), to_page=max(page_indices),
                   start_at=-1)
    # insert_pdf with a range includes pages in between — rebuild selectively instead
    out.close()

    out = fitz.open()
    for idx in page_indices:
        out.insert_pdf(src, from_page=idx, to_page=idx)
    src.close()

    pdf_bytes = out.tobytes()
    save_path.write_bytes(pdf_bytes)
    out.close()
    return pdf_bytes


def call_claude(pdf_bytes: bytes, prompt: str, client) -> str:
    """Send mini-PDF to Claude and return raw response text."""
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.standard_b64encode(pdf_bytes).decode(),
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    import anthropic
    from transformers import AutoModel

    load_dotenv()

    prompt_path = Path("home/j/jkuhlma1/2026_BA_Code/baselines/baseline_a_frontier_model/BaselineA-Prompt.txt")
    extraction_prompt = prompt_path.read_text()

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    query_embeddings = model.forward_queries([RETRIEVAL_QUERY], batch_size=1)

    pt_map = {p.stem: p for p in EMB_DIR.glob("*.pt")}

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        report_name = pdf_path.stem
        output_path = OUTPUT_DIR / f"{report_name}.json"

        if output_path.exists():
            print(f"[SKIP] {report_name}")
            continue

        if report_name not in pt_map:
            print(f"[WARN] no embedding for {report_name}")
            continue

        print(f"[RUN]  {report_name}")

        # Step 1 — Retrieval
        image_embeddings = torch.load(pt_map[report_name], weights_only=False, map_location="cpu")
        image_embeddings = [t.to("cuda") for t in image_embeddings]
        scores           = model.get_scores(query_embeddings, image_embeddings)  # [1, n_pages]
        pages            = select_pages(scores[0])
        print(f"         pages (0-idx): {pages}")

        # Step 2 — Extract selected pages as mini-PDF (saved + returned as bytes)
        retrieval_path = RESULT_DIR / pdf_path.name
        pdf_bytes      = extract_pages_as_pdf(pdf_path, pages, retrieval_path)

        # Step 3 — Extract with Claude
        raw = call_claude(pdf_bytes, extraction_prompt, client)

        # Step 4 — Persist
        output_path.write_text(json.dumps({
            "report":          report_name,
            "selected_pages":  pages,
            "retrieval_query": RETRIEVAL_QUERY,
            "raw_response":    raw,
        }, indent=2, ensure_ascii=False))
        print(f"         → {output_path}")


if __name__ == "__main__":
    main()