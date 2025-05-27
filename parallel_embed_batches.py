#!/usr/bin/env python3
import os
import subprocess
from multiprocessing import Process

def run_embedding(gpu_id: int, batch_idx: int):
    input_sdf = f"sdf_batches/batch_{batch_idx}.sdf"
    smiles_out = f"output/batch_{batch_idx}.csv"
    emb_out = f"output/batch_{batch_idx}.pt"
    log_file = f"logs/batch_{batch_idx}.log"

    # Assign specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python3", "build_db.py",
        "--source", "sdf",
        "--input_sdf", input_sdf,
        "--smiles_out", smiles_out,
        "--emb_out", emb_out
    ]

    with open(log_file, "w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        process.wait()

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    processes = []
    for i in range(8):  # For 8 batches and 8 GPUs
        p = Process(target=run_embedding, args=(i, i + 1))  # GPU i, batch_{i+1}
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
