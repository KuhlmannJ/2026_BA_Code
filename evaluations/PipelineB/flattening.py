import os
import json
import pandas as pd
from pathlib import Path

#### GLOBAL VARIABLES
SCOPES = ["scope_1", "scope_2_market_based", "scope_2_location_based", "scope_3"]

BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def flatten_json(filepath: Path) -> list[dict]:
    with open(filepath) as fh:
        data = json.load(fh)

    report_name = filepath.stem
    emissions   = data.get("emissions", {})
    rows = []

    for scope in SCOPES:
        for year, entries in emissions.get(scope, {}).items():
            for entry in entries:
                rows.append({
                    "report_name": report_name,
                    "scope":       scope,
                    "year":        year,
                    "value":       entry.get("value"),
                    "unit":        entry.get("unit"),
                    "label":       entry.get("label", ""),
                })

    return rows

input_dir = Path(BASE) / "PipelineB-Answers"
inputs  = sorted(p for p in input_dir.iterdir() if p.is_dir())
outputs = [p.with_suffix(".csv") for p in inputs]
## the above is esentially this:
# inputs  = [
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-30B-A3B-Thinking",
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-32B-Instruct",
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-32B-Thinking",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-32B-Instruct",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-30B-A3B-Thinking",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-32B-Thinking",
# ]
# outputs = [
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-32B-Instruct.csv",
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-30B-A3B-Thinking.csv",
#     Path(BASE) / "PipelineB-Answers/1st_Qwen3-VL-32B-Thinking.csv",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-32B-Instruct.csv",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-30B-A3B-Thinking.csv",
#     Path(BASE) / "PipelineB-Answers/2nd_Qwen3-VL-32B-Thinking.csv",
# ]

for i in range(len(inputs)):

    files = sorted(Path(inputs[i]).glob("*.json"))

    banner(f"Flattening {len(files)} JSONs => {outputs[i]}")

    all_rows = []
    errors   = []
    for f in files:
        try:
            all_rows.extend(flatten_json(f))
        except Exception as e:
            errors.append(f.name)
            print(f"  [WARN] {f.name}: {e}")

    df = pd.DataFrame(all_rows, columns=["report_name", "scope", "year", "value", "unit", "label"])
    df.to_csv(outputs[i], index=False)

    print(f"  Reports    : {df['report_name'].nunique()}")
    print(f"  Zeilen     : {len(df)}")
    print(f"  Fehler     : {len(errors)}")
    print(f"  Gespeichert: {outputs[i]}")