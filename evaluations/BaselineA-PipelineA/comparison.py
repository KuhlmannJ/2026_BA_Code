import pandas as pd

#### GLOBAL VARIABLES
MATCH_TOLERANCE = 0.01  # 1 % relative Abweichung gilt als Match    
    
#### INPUTS
ba = pd.read_csv("./baselineA.csv")
pa = pd.read_csv("./pipelineA.csv")

print(ba.info())
print(pa.info())

#### MERGE
df = pd.merge(
    ba[["report_name", "scope", "year", "value"]],
    pa[["report_name", "scope", "year", "value"]],
    on=["report_name", "scope", "year"],
    how="outer",
    suffixes=("_ba", "_pa"),
)

#### METRICS
df["rel_diff"] = (df["value_ba"] - df["value_pa"].abs()) / df["value_ba"].abs()
df["match"]    = df["rel_diff"] <= MATCH_TOLERANCE

df["source"]   = "both" # defaulting, may be overwritten with the following lines
df.loc[df["value_ba"].isna(), "source"] = "pipeline_only" #if "outer" generated NaN in ba => pipeline_only
df.loc[df["value_pa"].isna(), "source"] = "baseline_only"


# Zusammenfassung
both = df[df["source"] == "both"]
print(f"  No. of entries      : {len(df)}")
print(f"  In both.            : {len(both)}")
print(f"  Only Baseline A     : {(df["source"]=="baseline_only").sum()}")
print(f"  Only Pipeline A     : {(df["source"]=="pipeline_only").sum()}")
print(f"  Match (≤1%)         : {df["match"].sum()}/{len(both)}")
print(f"  Median Rel. Delta   : {both["rel_diff"].median():.3f}")



df.to_csv("A-A_comparison.csv", index=False)
print("Gespeichert: A-A_comparison.csv")

df["scope"] = df["scope"].str.replace("scope_1", "1", regex=False)
df["scope"] = df["scope"].str.replace("scope_2_location_based", "2lb", regex=False)
df["scope"] = df["scope"].str.replace("scope_2_market_based", "2mb", regex=False)
df["scope"] = df["scope"].str.replace("scope_3", "3", regex=False)



checklist = pd.read_csv("../../checklist.csv")
checklist["report_name"] = checklist["report_name"].str.replace(" ", "_", regex=False)
checklist["report_name"] = checklist["report_name"].str.replace(".pdf", "", regex=False)
checklist.to_csv("checklistWO_pdf.csv", index=False)

gs = pd.read_csv("checklistWO_pdf.csv")

gs["year"] = gs["year"].astype(str)
gs = gs.rename(columns={"value": "value_gs"})

print(gs.info())

df2 = pd.merge(
    df,
    gs[["report_name", "scope", "year", "value_gs"]],
    on=["report_name", "scope", "year"],
    how="left",
)

df2["match_gs_ba"] = df2["value_gs"] == df2["value_ba"]
df2["match_gs_pa"] = df2["value_gs"] == df2["value_pa"]
df2["match_gs_ba_pa"] = df2["match"] & df2["match_gs_ba"] & df2["match_gs_pa"]

df2.to_csv("A-A-G_comparison.csv", index=False)
print("Gespeichert: A-A-G_comparison.csv")

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