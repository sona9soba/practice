#!/usr/bin/env python3
# af3_batch_runner.py

import os, sys
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from JY import AF3Runner

def run_af3_batch(
    smiles_csv: str,
    uniprot_id: str,
    sequence: str,
    output_dir: str,
    num_batches: int = 1,
    run_now: bool = True
):
    runner = AF3Runner()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(smiles_csv)
    if "smiles" not in df.columns or "hash_id" not in df.columns:
        raise ValueError("CSV must contain both 'smiles' and 'hash_id' columns.")

    total = len(df)
    batch_size = (total + num_batches - 1) // num_batches

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total)
        batch_df = df.iloc[batch_start:batch_end]
        batch_dir = os.path.join(output_dir, f"input_{batch_idx+1}")
        os.makedirs(batch_dir, exist_ok=True)

        for _, row in batch_df.iterrows():
            smiles = row["smiles"]
            hash_id = row["hash_id"]
            jobname = f"{uniprot_id}_{hash_id}"
            try:
                print(f"🧬 Preparing: {jobname} in {batch_dir}")
                json_path = runner.prepare_input(
                    af_input_dir=batch_dir,
                    jobname=jobname,
                    smiles=smiles,
                    uniprot_id=uniprot_id,
                    protein_sequence=sequence
                )
                if run_now:
                    print(f"🚀 Running AF3 for: {jobname}")
                    runner.run_af3(input_json=json_path)
            except Exception as e:
                print(f"❌ Failed on {jobname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AlphaFold3 for multiple ligands with one protein")
    parser.add_argument("--ligand_csv", required=True, help="CSV file with 'smiles' and 'hash_id' columns")
    parser.add_argument("--uniprot_id", required=True, help="UniProt ID of the protein")
    parser.add_argument("--sequence", required=True, help="Protein sequence string")
    parser.add_argument("--output_dir", default="af_inputs", help="Base output directory")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batch directories to divide into")
    parser.add_argument("--no_run", action="store_true", help="Only prepare inputs, don't execute AF3")
    args = parser.parse_args()

    run_af3_batch(
        smiles_csv=args.ligand_csv,
        uniprot_id=args.uniprot_id,
        sequence=args.sequence,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        run_now=not args.no_run
    )


"""
python3 af3_batch_runner.py \
  --ligand_csv ligands_r4.csv \
  --uniprot_id P9WIL5 \
  --sequence "MTIPAFHPGELNVYSAPGDVADVSRALRLTGRRVMLVPTMGALHEGHLALVRAAKRVPGSVVVVSIFVNPMQFGAGEDLDAYPRTPDDDLAQLRAEGVEIAFTPTTAAMYPDGLRTTVQPGPLAAELEGGPRPTHFAGVLTVVLKLLQIVRPDRVFFGEKDYQQLVLIRQLVADFNLDVAVVGVPTVREADGLAMSSRNRYLDPAQRAAAVALSAALTAAAHAATAGAQAALDAARAVLDAAPGVAVDYLELRDIGLGPMPLNGSGRLLVAARLGTTRLLDNIAIEIGTFAGTDRPDGYRAILESHWRN" \
  --num_batches 7 \
  --no_run \
  --output_dir /home/ubuntu/af_inputs

"""