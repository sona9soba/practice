# merge_csvs.py
import pandas as pd
import glob

csv_files = sorted(glob.glob("output/batch_*.csv"))
dfs = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv("output/merged_smiles.csv", index=False)

print(f"✅ Merged {len(csv_files)} CSVs into output/merged_smiles.csv")
