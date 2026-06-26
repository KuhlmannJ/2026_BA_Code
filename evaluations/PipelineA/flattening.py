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
    emissions = data.get("emissions", {})
    rows = []

    # Iterate over the scopes actually present in the JSON so typos are visible
    for scope, years in emissions.items():

        # Both errors get catched at funcion call, so the script continues bur "Fehler : " raises.
        if not isinstance(years, dict):
            raise ValueError(f"{report_name}: expected dict for scope '{scope}', got {type(years).__name__}")

        if scope not in SCOPES:
            if scope =="scope_2":
                scope = "scope_2_location_based" # Three reports only got back with "scope_2": This meant location based
            else:
                raise ValueError(f"{report_name}: unexpected scope '{scope}'")

        for year, entries in years.items():
            for entry in entries or []:
                rows.append({
                    "report_name": report_name,
                    "scope":       scope,
                    "year":        year,
                    "value":       entry.get("value"),
                    "unit":        entry.get("unit"),
                    "label":       entry.get("label", ""),
                })

    return rows

input_dir = Path(BASE)
inputs  = sorted(p for p in input_dir.iterdir() if p.is_dir())
outputs = [p.with_suffix(".csv") for p in inputs]

total_errors = []

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
            total_errors.append(errors)
            print(f"  [WARN] {f.name}: {e}")

    df = pd.DataFrame(all_rows, columns=["report_name", "scope", "year", "value", "unit", "label"])
    df.to_csv(outputs[i], index=False)

    
    print(f"  Reports    : {df['report_name'].nunique()}")
    print(f"  Zeilen     : {len(df)}")
    print(f"  Fehler     : {len(errors)}")
    print(f"  Gespeichert: {outputs[i]}")
    
print()
print(f"Total Errors: {len(total_errors)}")

print("[WARN] ERRORS DETECTED") if len(total_errors) > 0 else print("[OK] No errors detected.")