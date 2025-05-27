#!/usr/bin/env python3
from rdkit import Chem
import os

def split_sdf(input_sdf: str, output_prefix: str, num_batches: int):
    suppl = Chem.SDMolSupplier(input_sdf)
    mols = [mol for mol in suppl if mol is not None]
    total = len(mols)
    batch_size = (total + num_batches - 1) // num_batches  # ceil division

    os.makedirs(output_prefix, exist_ok=True)

    for i in range(num_batches):
        batch_mols = mols[i * batch_size:(i + 1) * batch_size]
        writer = Chem.SDWriter(os.path.join(output_prefix, f"batch_{i+1}.sdf"))
        for mol in batch_mols:
            writer.write(mol)
        writer.close()
        print(f"✅ Saved batch_{i+1}.sdf with {len(batch_mols)} molecules.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split an SDF into multiple batches.")
    parser.add_argument("--input_sdf", required=True, help="Path to input SDF file")
    parser.add_argument("--output_dir", default="sdf_batches", help="Directory to save batches")
    parser.add_argument("--num_batches", type=int, default=8, help="Number of batches")
    args = parser.parse_args()

    split_sdf(args.input_sdf, args.output_dir, args.num_batches)



"""
python3 split_sdf.py --input_sdf mce_leadlike.sdf --output_dir sdf_batches --num_batches 8

"""