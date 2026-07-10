import os
import pandas as pd
from pathlib import Path
#### GLOBAL VARIABLES
BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located
AA_DIR = Path(BASE) / "../Baseline-PipelineA"  # inputs/outputs live in the Baseline-PipelineA folder
MATCH_TOLERANCE = 0.01  # 1 % relative Abweichung gilt als Match

#### INPUTS
ba = pd.read_csv(AA_DIR / "baseline.csv")
pa = pd.read_csv(AA_DIR / "pipelineA.csv")

print(ba.info())
print(pa.info())

#### MERGE
AA = pd.merge(
    ba[["report_name", "scope", "year", "value"]],
    pa[["report_name", "scope", "year", "value"]],
    on=["report_name", "scope", "year"],
    how="outer",
    suffixes=("_ba", "_pa"),
)

#### METRICS
AA["rel_diff"] = (AA["value_ba"] - AA["value_pa"].abs()) / AA["value_ba"].abs()
AA["match"]    = AA["rel_diff"] <= MATCH_TOLERANCE

AA["source"]   = "both" # defaulting, may be overwritten with the following lines
AA.loc[AA["value_ba"].isna(), "source"] = "pipeline_only" #if "outer" generated NaN in ba => pipeline_only
AA.loc[AA["value_pa"].isna(), "source"] = "baseline_only"


# Zusammenfassung
both = AA[AA["source"] == "both"]
print(f"  No. of entries      : {len(AA)}")
print(f"  In both.            : {len(both)}")
print(f"  Only Baseline       : {(AA["source"]=="baseline_only").sum()}")
print(f"  Only Pipeline A     : {(AA["source"]=="pipeline_only").sum()}")
print(f"  Match (≤1%)         : {AA["match"].sum()}/{len(both)}")
print(f"  Median Rel. Delta   : {both["rel_diff"].median():.3f}")



AA.to_csv(AA_DIR / "A-A_comparison.csv", index=False)
print("Gespeichert: A-A_comparison.csv")

AA["scope"] = AA["scope"].str.replace("scope_1", "1", regex=False)
AA["scope"] = AA["scope"].str.replace("scope_2_location_based", "2lb", regex=False)
AA["scope"] = AA["scope"].str.replace("scope_2_market_based", "2mb", regex=False)
AA["scope"] = AA["scope"].str.replace("scope_3", "3", regex=False)



#### LOAD GOLD STANDARD (gs_slim.json)

gs = pd.read_json(Path(BASE) / "../gs_slim.json")

# gs_slim holds one row per (report_name, year, scope) even where no value was
# reported — those empty cells exist for recall, not for value comparison.
gs = gs[gs["value"].notna()]

gs["year"] = gs["year"].astype(str)
gs = gs.rename(columns={"value": "value_gs"})

# print(gs.info())

AAG = pd.merge(
    AA,
    gs[["report_name", "scope", "year", "value_gs"]],
    on=["report_name", "scope", "year"],
    how="left",
)

#### COMPARE Base- and Pipeline with GS

AAG["match_gs_ba"] = AAG["value_gs"] == AAG["value_ba"]
AAG["match_gs_pa"] = AAG["value_gs"] == AAG["value_pa"]
AAG["match_gs_ba_pa"] = AAG["match"] & AAG["match_gs_ba"] & AAG["match_gs_pa"]

AAG.to_csv(AA_DIR / "A-A-G_comparison.csv", index=False)
print("Gespeichert: A-A-G_comparison.csv")

### Problem analysis

bad_reports = AAG.loc[AAG["match"] != "both", "report_name"].unique()
AAG_bad = AAG[AAG["report_name"].isin(bad_reports)]

AAG_bad.to_csv(AA_DIR / "A-A-G_badRows.csv", index=False)



#### SPELLING CANDIDATES (none found anymore)

# Find unmatched rows from both sides that share scope + year + value
# to catch report_name spelling errors
# ba_only = df[df["source"] == "baseline_only"][["report_name", "scope", "year", "value_ba"]].rename(columns={"report_name": "report_name_ba", "value_ba": "value"})
# pa_only = df[df["source"] == "pipeline_only"][["report_name", "scope", "year", "value_pa"]].rename(columns={"report_name": "report_name_pa", "value_pa": "value"})

# spelling_candidates = pd.merge(
#     ba_only,
#     pa_only,
#     on=["scope", "year", "value"],
#     how="inner",
# )
# spelling_candidates = spelling_candidates[spelling_candidates["report_name_ba"] != spelling_candidates["report_name_pa"]]

# print(f"\n  Spelling candidates : {len(spelling_candidates)}")
# if not spelling_candidates.empty:
#     print(spelling_candidates[["scope", "year", "value", "report_name_ba", "report_name_pa"]].to_string(index=False))