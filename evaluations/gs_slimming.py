import os
import json
import pandas as pd
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__)) # sets "BASE" to directory this .py is located

### Listing all downloadable reports
#reports_downloaded = {p.name for p in Path("localdata/all_esg_reports").glob("*.pdf")}
reports_downloaded = {p.name for p in (Path(BASE) / "../localdata/all_esg_reports").glob("*.pdf")}



### Loading Gold_Standard in full
#gs = pd.read_csv("evaluations/gold_standard.csv")
gs = pd.read_csv((Path(BASE) / "../evaluations/gold_standard.csv"))

# Finding wrong field name in gs and mapping it
print("Mismatches:", reports_downloaded - set(gs["report_name"]))
gs["report_name"] = gs["report_name"].replace("viacomcbs_2020_report.pdf", "ViacomCBS_ESG Report_2020-2021_vFINAL.pdf")

# Correcting wrong page number for 1 report in GS
gs.loc[
    (gs["report_name"] == "sumitomo corporation_2021_report.pdf") & 
    (gs["page"].notna()) & 
    (gs["page"] == "121"), 
    "page"
] = "124"

# Defining Status Column for each report, beginning with "notavail"
gs["status"] = gs["report_name"].apply(lambda r: None if r in reports_downloaded else "notavail")

# Stipping unnecessary columns
toKeep = ["report_name", "year", "scope", "page", "value", "unit", "unit_normalized", "metric_name", "status"]
gs_slim = gs[toKeep]
gs_slim = gs_slim[gs_slim["status"]!="notavail"]



print()
print(f"No. of gold_standard reports: {gs['report_name'].nunique()}")
print()
print(f"No. of downloadable reports: {len(reports_downloaded)}")

##########################################
### NOW Manipulating gs_slim

## Mapping gs_slim for better analysis capabilities down the road
gs_slim["report_name"]  = gs_slim["report_name"].str.replace(".pdf", "")
gs_slim = gs_slim.rename(columns={"metric_name": "label"})

# Sorting rows like VSC does it with JSON
gs_slim["_sort"] = gs_slim["report_name"].str.lower() # need to circumvent ASCII case-sensitive sorting
gs_slim = gs_slim.sort_values(["_sort", "scope", "year"], ignore_index=True).drop(columns="_sort")


##########################################
### Evaluation which scopes for which reports are present and defining "status" for each report


# Categorizing reports by scope coverage
# Counting how many non-null values in ["values"] are present for each combination of ["report_name", "scope"]
# Per report one row and writes scopes in colums to concante later
scope_counts = (
    gs_slim.groupby(["report_name", "scope"])["value"]
    .apply(lambda x: x.notna().sum())
    .unstack(fill_value=0)
)

# Returns categories for each scope for every report
# "useless" if all are None
# "complete" if all 4 Scopes are avail
# "partial" else
def categorize(row):
    ORDER = ["1", "2lb", "2mb", "3"]
    present = [s for s in ORDER if row.get(s, 0) > 0]
    if not present:
        return "useless", []
    return ("complete" if len(present) == 4 else "partial"), present

# Applying func `categorize` on every created rows on scope_coounts, "result_type="expand"" brings ouput into two columns
scope_counts[["status", "scopes_present"]] = scope_counts.apply(categorize, axis=1, result_type="expand")

# Merging those resilts back onto gs_slim
gs_slim = gs_slim.drop(columns="status").merge(scope_counts[["status", "scopes_present"]], on="report_name")


##########################################
### Evaluation which years for which reports are filled with data 

years_present = (
    gs_slim[gs_slim["value"].notna()]
    .groupby("report_name")["year"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .rename("years_present")
)

gs_slim = gs_slim.merge(years_present, on="report_name")



# ##########################################
# ### Prepare export-friendly scope columns

# # Ensure `scopes_present` is a list for downstream processing
# gs_slim["scopes_present"] = gs_slim["scopes_present"].apply(lambda x: x if isinstance(x, (list, tuple)) else [])
# # JSON-serializable column for CSV/transfer and a human-readable string column
# gs_slim["scopes_present_export"] = gs_slim["scopes_present"].apply(json.dumps)

##########################################
### Saving gs_slim to JSON
gs_slim.to_json(Path(BASE) / "gs_slim.json",index=False, orient="records", indent=4)

print("="*60)
print("="*60)
print()
print(f"No. of reports in gs_slim: {gs_slim['report_name'].nunique()}")
print()
print()
print(gs_slim.drop_duplicates("report_name")["status"].value_counts())
print()
print(gs_slim.drop_duplicates("report_name")["scopes_present"].value_counts(dropna=False))