#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel
import selfies as sf
import pandas as pd
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Optional FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

tokenizer = AutoTokenizer.from_pretrained("ibm-research/materials.selfies-ted")
model = AutoModel.from_pretrained("ibm-research/materials.selfies-ted")
model.eval()

_db_smiles: List[str] = []
_db_matrix: torch.Tensor
_db_hash_map: Dict[str, str] = {}
_faiss_index = None
_use_faiss = False

def featurize_selfies(smiles: str) -> torch.Tensor:
    try:
        selfies_str = sf.encoder(smiles)
        inputs = tokenizer(selfies_str, return_tensors="pt", add_special_tokens=True)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return (summed / counts).squeeze(0)
    except Exception as e:
        print(f"[WARN] Failed to featurize: {smiles} — {type(e).__name__}: {e}")
        return None

@lru_cache(maxsize=None)
def featurize_selfies_cached(smiles: str) -> torch.Tensor:
    return featurize_selfies(smiles)

def load_database(smiles_file: str, emb_file: str, faiss_index: str = None):
    global _db_smiles, _db_matrix, _db_hash_map, _faiss_index, _use_faiss

    print(f"📂 Loading SMILES from: {smiles_file}")
    smiles_df = pd.read_csv(smiles_file)
    if "smiles" not in smiles_df.columns or "hash_id" not in smiles_df.columns:
        raise ValueError("❌ SMILES DB must contain 'smiles' and 'hash_id' columns.")
    _db_smiles[:] = smiles_df["smiles"].tolist()
    _db_hash_map = dict(zip(smiles_df["smiles"], smiles_df["hash_id"]))
    print(f"✅ Loaded {len(_db_smiles)} SMILES.")

    print(f"📂 Loading embeddings from: {emb_file}")
    _db_matrix = torch.load(emb_file)
    print(f"✅ Loaded embeddings with shape {_db_matrix.shape}")

    if faiss_index:
        if not FAISS_AVAILABLE:
            raise ImportError("❌ FAISS is not installed.")
        if not os.path.exists(faiss_index):
            raise FileNotFoundError(f"❌ FAISS index not found: {faiss_index}")
        print(f"📂 Loading FAISS index: {faiss_index}")
        _faiss_index = faiss.read_index(faiss_index)
        _use_faiss = True
        print(f"✅ FAISS index loaded: {_faiss_index.ntotal} vectors")

def load_used_hashes(af_inputs_dir: str) -> set:
    used_hashes = set()
    if not os.path.exists(af_inputs_dir):
        print(f"[WARN] af_inputs directory not found: {af_inputs_dir}")
        return used_hashes

    for subdir in os.listdir(af_inputs_dir):
        full_path = os.path.join(af_inputs_dir, subdir)
        if os.path.isdir(full_path):
            for fname in os.listdir(full_path):
                if "_" in fname:
                    parts = fname.split("_", 1)
                    if len(parts) > 1:
                        hash_part = parts[1].split(".")[0]
                        used_hashes.add(hash_part)
    print(f"🧾 Loaded {len(used_hashes)} used hash_ids from {af_inputs_dir}")
    return used_hashes

def get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return ""

def find_similar(smiles: str, top_k: int = 10, exclude_top_percent: float = 0.0, exclude_hashes: set = None) -> List[Tuple[str, float]]:
    q_emb = featurize_selfies_cached(smiles)
    if q_emb is None:
        return []

    if _use_faiss:
        q_np = q_emb.numpy().astype("float32").reshape(1, -1)
        D, I = _faiss_index.search(q_np, top_k * 5)
        results = []
        for j, i in enumerate(I[0]):
            if i == -1:
                continue
            smi = _db_smiles[i]
            hash_id = _db_hash_map.get(smi, "NA")
            if exclude_hashes and hash_id in exclude_hashes:
                continue
            sim_score = float(1 - D[0][j])
            results.append((smi, sim_score))
        return results
    else:
        sims = F.cosine_similarity(q_emb.unsqueeze(0), _db_matrix)
        if exclude_top_percent > 0.0:
            threshold_idx = int(len(sims) * exclude_top_percent)
            sims_sorted, indices_sorted = sims.sort(descending=True)
            sims = sims_sorted[threshold_idx:]
            indices = indices_sorted[threshold_idx:]
        else:
            sims, indices = sims.sort(descending=True)

        results = []
        for j, i in enumerate(indices):
            smi = _db_smiles[i]
            hash_id = _db_hash_map.get(smi, "NA")
            if exclude_hashes and hash_id in exclude_hashes:
                continue
            results.append((smi, float(sims[j])))
        return results

def main():
    parser = argparse.ArgumentParser(description="SELFIES-TED similarity search with scaffold filtering and global logging")
    parser.add_argument("--input_csv", required=True, help="CSV with 'smiles' column")
    parser.add_argument("--output_csv", required=True, help="Where to save results")
    parser.add_argument("--smiles_db", default="chembl_selfies_smiles.csv")
    parser.add_argument("--emb_db", default="chembl_selfies_emb.pt")
    parser.add_argument("--faiss_index", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--exclude_percent", type=float, default=0.0)
    parser.add_argument("--af_inputs_dir", type=str, default=None)
    args = parser.parse_args()

    load_database(args.smiles_db, args.emb_db, faiss_index=args.faiss_index)

    # Exclude hash_ids from af_inputs and global log
    exclude_hashes = set()
    if args.af_inputs_dir:
        exclude_hashes.update(load_used_hashes(args.af_inputs_dir))
    global_log_path = "used_hash_ids_global.txt"
    if os.path.exists(global_log_path):
        with open(global_log_path, "r") as f:
            exclude_hashes.update(line.strip() for line in f)

    df = pd.read_csv(args.input_csv)
    if "smiles" not in df.columns:
        raise ValueError("❌ Input CSV must contain a 'smiles' column.")

    all_results = []
    global_used_scaffolds = set()

    for i, row in df.iterrows():
        query = row["smiles"]
        print(f"🔍 Query {i+1}/{len(df)}: {query}")

        results = find_similar(query, top_k=args.top_k * 5, exclude_top_percent=args.exclude_percent, exclude_hashes=exclude_hashes)
        used_scaffolds = set()
        fallback_results = []

        for smi, score in results:
            hash_id = _db_hash_map.get(smi, "NA")
            scaffold = get_scaffold(smi)
            record = {
                "query": query,
                "smiles": smi,
                "cosine_similarity": score,
                "hash_id": hash_id,
                "scaffold": scaffold
            }
            if scaffold and scaffold not in used_scaffolds and scaffold not in global_used_scaffolds:
                all_results.append(record)
                used_scaffolds.add(scaffold)
                global_used_scaffolds.add(scaffold)
                exclude_hashes.add(hash_id)
            else:
                fallback_results.append(record)

            if len(used_scaffolds) >= args.top_k:
                break

        if len(used_scaffolds) < args.top_k:
            needed = args.top_k - len(used_scaffolds)
            all_results.extend(fallback_results[:needed])

    # Save result CSV
    out_df = pd.DataFrame(all_results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✅ Results saved to {args.output_csv}")

    # Update global hash_id log
    used_hash_ids = {r["hash_id"] for r in all_results if r["hash_id"] != "NA"}
    if os.path.exists(global_log_path):
        with open(global_log_path, "r") as f:
            existing_hashes = set(line.strip() for line in f)
    else:
        existing_hashes = set()
    all_logged = existing_hashes.union(used_hash_ids)
    with open(global_log_path, "w") as f:
        for h in sorted(all_logged):
            f.write(h + "\n")
    print(f"🧾 Global used_hash_ids saved to: {global_log_path}")

if __name__ == "__main__":
    main()

"""
python3 search_selfies_ted.py \
  --input_csv ligands_r3_docking_smiles.csv \
  --output_csv ligands_r4.csv \
  --smiles_db mce_leadlike.csv \
  --emb_db mce_leadlike.pt \
  --faiss_index faiss_hnsw.index \
  --top_k 1500 \
  --af_inputs_dir /home/ubuntu/af_inputs

""]
