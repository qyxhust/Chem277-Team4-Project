from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

rawdata = pd.read_excel("41591_2025_3834_MOESM3_ESM.xlsx", sheet_name="SuppTbl5")
rawdata = rawdata["Unnamed: 4"].dropna().drop_duplicates().astype(str)
#print(rawdata)

aliases = "9606.protein.aliases.v12.0.txt"
links  = "9606.protein.links.full.v12.0.txt"

aliases = pd.read_csv(
        aliases,
        sep="\t",
        header=None,
        names=["string_id", "alias", "source"],
        encoding="utf-8"
    )

# Basic cleanup
aliases["string_id"] = aliases["string_id"].astype(str).str.strip()
aliases["alias"]     = aliases["alias"].astype(str).str.strip()
aliases["source"]    = aliases["source"].astype(str).str.strip()

# Filter for relevant source
sub = aliases[aliases["alias"].isin(rawdata)].copy()

out = (sub.rename(columns={"alias":"UniProt", "string_id":"ENSP"})
          [["UniProt","ENSP","source"]]
          .sort_values(["UniProt","ENSP","source"])
          .reset_index(drop=True))

out.to_csv("namelink.csv", index=False)
print(f"[OK] saved {len(out)} rows -> namelink")

unmatched = rawdata[~rawdata.isin(out["UniProt"])]
if len(unmatched) > 0:
    unmatched.to_frame(name="UniProt").to_csv("uniprot_unmatched.csv", index=False)
    print(f"[Info] {len(unmatched)} UniProt had no mapping. See uniprot_unmatched.csv")
