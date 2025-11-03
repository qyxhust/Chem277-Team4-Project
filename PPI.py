import pandas as pd
import numpy as np
import os

MIN_SCORE = 700
TOPK    = 20  
OUT_DIR = "ppi_output"

linkpath  = "9606.protein.links.full.v12.0.txt"

links= pd.read_csv(linkpath, delim_whitespace=True, header=0, encoding="utf-8")

#print(links.columns.tolist())

name = pd.read_csv("namelink.csv")

V = set(name["ENSP"].tolist())
print(f"[Map] ENSP nodes from mapping: {len(V)}")


df = links[links["protein1"].isin(V) & links["protein2"].isin(V)].copy()
print(f"[Links] edges touching V before threshold/topk: {len(df)}")

if "combined_score" in df.columns and MIN_SCORE is not None:
    df = df[df["combined_score"] >= int(MIN_SCORE)].copy()
    print(f"[Links] after score >= {MIN_SCORE}: {len(df)}")

if TOPK is not None and len(df) > 0:
    left = (df.sort_values(["protein1","combined_score"], ascending=[True, False])
              .groupby("protein1", as_index=False).head(TOPK))
    swap = df.rename(columns={"protein1":"protein2","protein2":"protein1"})
    right = (swap.sort_values(["protein1","combined_score"], ascending=[True, False])
                 .groupby("protein1", as_index=False).head(TOPK)
                 .rename(columns={"protein1":"protein2","protein2":"protein1"}))
    df = pd.concat([left, right], ignore_index=True).drop_duplicates(subset=["protein1","protein2"])
    print(f"[Links] after symmetric top-{TOPK}: {len(df)}")


nodes = sorted(set(df["protein1"]).union(set(df["protein2"])))
idx = {n:i for i,n in enumerate(nodes)}
# 把 UniProt 放回去便于对齐
rep = name[["ENSP", "UniProt"]].drop_duplicates().set_index("ENSP")
nodes_out = pd.DataFrame({
    "node_id": nodes,
    "index": [idx[n] for n in nodes],
    "UniProt": [rep.loc[n, "UniProt"] if n in rep.index else np.nan for n in nodes],
})
nodes_out.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)

if len(df) > 0:
    if "combined_score" in df.columns:
        smax = max(df["combined_score"].max(), 1.0)
        edges_out = pd.DataFrame({
            "src": df["protein1"].map(idx).astype(int),
            "dst": df["protein2"].map(idx).astype(int),
            "weight": (df["combined_score"]/smax).astype(float),
            "combined_score": df["combined_score"].astype(float)
        })
    else:
        edges_out = pd.DataFrame({
            "src": df["protein1"].map(idx).astype(int),
            "dst": df["protein2"].map(idx).astype(int),
            "weight": 1.0
        })
else:
    edges_out = pd.DataFrame(columns=["src","dst","weight","combined_score"])
edges_out.to_csv(os.path.join(OUT_DIR, "edges.csv"), index=False)


ensps_in_links = set(links["protein1"]).union(set(links["protein2"]))
covered = len(V & ensps_in_links)
print(f"[Report] nodes exported: {len(nodes_out)} | edges exported: {len(edges_out)}")
print(f"[Report] of {len(V)} mapped ENSPs, {covered} appear in links file.")
print(f"[OK] Saved to {OUT_DIR}/nodes.csv and {OUT_DIR}/edges.csv")
