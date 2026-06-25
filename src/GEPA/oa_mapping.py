import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

SCOPES = ["scope_1", "scope_2_market_based", "scope_2_location_based", "scope_3"]
SCOPE_MAPPING = {
    "scope_1":               "1",
    "scope_2_location_based":"2lb",
    "scope_2_market_based":  "2mb",
    "scope_3":               "3",
}
MERGE_ON  = ["report_name", "scope", "year"]
AGG_COLS  = ["value", "unit", "label"]
SUFFIX    = "_instr"


def _flatten_json(filepath: Path) -> list[dict]:
    with open(filepath) as fh:
        data = json.load(fh)
    report_name = filepath.stem
    rows = []
    for scope, years in data.get("emissions", {}).items():
        if not isinstance(years, dict):
            raise ValueError(f"{report_name}: expected dict for scope '{scope}', got {type(years).__name__}")
        if scope not in SCOPES:
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

# RegEx Normalization for mapping the years
def _normalize_year(raw: str, years_present: set[int] | None = None) -> int | None:
    label = str(raw).strip().upper().removeprefix("FY").strip()
    if re.fullmatch(r"\d{4}", label):
        return int(label)
    if re.fullmatch(r"\d{2}", label):
        return 2000 + int(label)
    m = re.fullmatch(r"(\d{4})/(\d{1,4})", label)
    if m:
        left, right = m.groups()
        if len(right) == 4:
            return int(right)
        candidates = {int(left), int(left) + 1}
        if years_present:
            hit = candidates & years_present
            if len(hit) == 1:
                return hit.pop()
        return int(left)
    return None



# Flatten JSONs from run_dir, normalize years, and merge onto the gold standard.
# Returns a DataFrame structured like gs_extractions_raw_ynorm but with only _instr columns.
def map_to_goldstandard(run_dir: Path, gs_path: Path) -> pd.DataFrame:

    #### Getting GoldStandard ready
    gs = pd.read_json(gs_path)
    gs["year"] = gs["year"].astype(int) #As we normalize the years
    
    #### Getting extraction ready (reading jsons)
    all_rows = []
    errors   = []
    for f in sorted(run_dir.glob("*.json")):
        try:
            all_rows.extend(_flatten_json(f))
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    if errors:
        print(f"[WARN] {len(errors)} flatten errors:")
        for err in errors:
            print(f"  {err}")

    
    #### Getting extraction ready (putting jsons into dataFrame)
    extraction = pd.DataFrame(all_rows, columns=["report_name", "scope", "year", "value", "unit", "label"])

    extraction["scope"] = extraction["scope"].replace(SCOPE_MAPPING)
    extraction["value"] = pd.to_numeric(
        extraction["value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    years_in_extraction = set(extraction["year"].dropna().unique().tolist())
    extraction["year"] = extraction["year"].apply(_normalize_year, years_present=years_in_extraction)

    agg = (
        extraction.groupby(MERGE_ON)[AGG_COLS]
        .agg(list)
        .reset_index()
        .rename(columns={col: f"{col}{SUFFIX}" for col in AGG_COLS})
    )

    # Slimming down GoldStandard even furhter to just include the reports that were extracted (e.g. via a smaller traning set for GEPA)
    extraced_reports = set(agg["report_name"].unique())
    gs = gs[gs["report_name"].isin(extraced_reports)]
    
    merged = pd.merge(gs, agg, on=MERGE_ON, how="left") #"Left" to map extraction onto Gold-Standard

    for col in [f"{c}{SUFFIX}" for c in AGG_COLS]:
        merged[col] = merged[col].apply(lambda x: np.nan if x is None else x)

    return merged