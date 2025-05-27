#!/usr/bin/env python3
import os
import torch
import hashlib
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModel
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

# ✅ Load model/tokenizer once (global)
tokenizer = AutoTokenizer.from_pretrained("ibm-research/materials.selfies-ted")
model = AutoModel.from_pretrained("ibm-research/materials.selfies-ted").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def featurize_batch(smiles_batch: List[str]) -> List[torch.Tensor]:
    """
    Featurize a batch of SMILES into embeddings using SELFIES-TED.
    """
    try:
        selfies_batch = [sf.encoder(smi) for smi in smiles_batch]
        inputs = tokenizer(selfies_batch, return_tensors="pt", padding=True, truncation=True)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return [(summed[i] / counts[i]).cpu() for i in range(len(smiles_batch))]
    except Exception as e:
        print(f"[WARN] Batch featurization failed: {e}")
        return [None] * len(smiles_batch)

def smiles_from_sdf(sdf_path: str) -> List[str]:
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"❌ File not found: {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path)
    smiles_list = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            raw_smiles = mol.GetProp("SMILES")
            mol_from_smi = Chem.MolFromSmiles(raw_smiles)
            if mol_from_smi is None:
                continue
            canonical = Chem.MolToSmiles(mol_from_smi, canonical=True)
            smiles_list.append(canonical)
        except Exception as e:
            print(f"[WARN] Could not read SMILES from record: {e}")
    return smiles_list

def generate_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()

def resume_batch_save(smiles_all: List[str], smiles_out: str, emb_out: str, batch_size=50):
    os.makedirs(os.path.dirname(smiles_out) or ".", exist_ok=True)

    done_smiles = set()
    all_hashes, all_smiles, all_embeddings = [], [], []

    # Resume support
    if os.path.exists(smiles_out) and os.path.exists(emb_out):
        df = pd.read_csv(smiles_out)
        done_smiles = set(df["smiles"])
        all_hashes = list(df["hash_id"])
        all_smiles = list(df["smiles"])
        all_embeddings = torch.load(emb_out).tolist()

    for i in tqdm(range(0, len(smiles_all), batch_size), desc="📦 Processing in batches"):
        batch = smiles_all[i:i+batch_size]
        batch = [smi for smi in batch if smi not in done_smiles]
        if not batch:
            continue

        batch_embeddings = featurize_batch(batch)
        for smi, emb in zip(batch, batch_embeddings):
            if emb is not None:
                all_smiles.append(smi)
                all_hashes.append(generate_hash(smi))
                all_embeddings.append(emb.tolist())

        # Save every batch
        pd.DataFrame({"hash_id": all_hashes, "smiles": all_smiles}).to_csv(smiles_out, index=False)
        torch.save(torch.tensor(all_embeddings), emb_out)

    print(f"✅ Finished: {len(all_smiles)} molecules saved to {smiles_out}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sdf"], required=True)
    parser.add_argument("--input_sdf", required=True)
    parser.add_argument("--smiles_out", default="output.csv")
    parser.add_argument("--emb_out", default="output.pt")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    smiles = smiles_from_sdf(args.input_sdf)
    resume_batch_save(smiles, args.smiles_out, args.emb_out, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
