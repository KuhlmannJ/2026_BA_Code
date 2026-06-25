import gepa.optimize_anything as oa
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig
from pathlib import Path

from oa_evaluate import evaluate

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI
import os

# ── Paths
_HERE            = Path(__file__).parent
SEED_PROMPT_PATH = _HERE / "Query0_Extraction.txt"
RESULT_PATH      = _HERE / "oa_result.txt"

# ── Reflection model (UniGPT, fastest option)
REFLECTION_LM = "openai/gpt-oss-120b"

OBJECTIVE = (
    "Optimize a system prompt for a Vision Language Model (VLM) that extracts GHG emission values from corporate ESG/sustainability PDF reports. "
    "The prompt must instruct the model to identify Scope 1, Scope 2 location-based, Scope 2 market-based, "
    "and Scope 3 emissions for every reported year, along with their numeric values, units and, if necessary, labels. "
    "There may be more than one value for the same scope and year. The goal is to only extract the total value and nothing else. "
    "If it is ambiguous or simply not the total value, the evaluator will say, that there is no value to extract. "
    "Output must be a valid JSON object exactly matching the schema from the seed prompt! "
    "There are multiples years per report! Therefore keep the JSON in its orginial form and allow for multiple years per scope."
    "The extracted years get deterministically normalized via RegEx within the evaluator. "
    "Maximize the hit rate: the fraction of Gold-Standard values that are present in the extraction. "
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

seed = SEED_PROMPT_PATH.read_text()
print(f"Seed prompt loaded from: {SEED_PROMPT_PATH}")


result = oa.optimize_anything(
    seed_candidate=seed,
    evaluator=evaluate,
    objective=OBJECTIVE,
    config=GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=30,  # each call = one full extraction run, adjust to time budget
        ),
        reflection=ReflectionConfig(
            reflection_lm=REFLECTION_LM,
        ),
    ),
)

print("\n" + "=" * 60)
print(f"  Best score: {result.val_aggregate_scores[result.best_idx]:.4f}")
print("=" * 60)
print(result.best_candidate)

RESULT_PATH.write_text(result.best_candidate)
print(f"\nSaved to: {RESULT_PATH}")