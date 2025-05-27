#!/bin/bash

# Ensure required directories exist
mkdir -p output logs
# Launch 8 batches in parallel, one per GPU (GPU 0 to 7)
for i in {1..8}; do
  GPU_ID=$((i - 1))
  INPUT_SDF="sdf_batches/batch_${i}.sdf"
  SMILES_OUT="output/batch_${i}.csv"
  EMB_OUT="output/batch_${i}.pt"
  LOG_FILE="logs/batch_${i}.log"

  echo "🚀 Launching batch $i on GPU $GPU_ID..."
  
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python3 build_db.py \
    --source sdf \
    --input_sdf "$INPUT_SDF" \
    --smiles_out "$SMILES_OUT" \
    --emb_out "$EMB_OUT" \
    --batch_size 256 \
    > "$LOG_FILE" 2>&1 &

  sleep 2  # small delay to avoid contention during loading
done

# Wait for all parallel jobs to complete
wait
echo "✅ All batches completed."
