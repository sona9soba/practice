# merge_pts.py
import torch
import glob

pt_files = sorted(glob.glob("output/batch_*.pt"))
embeddings = [torch.load(f) for f in pt_files]
merged_tensor = torch.cat(embeddings, dim=0)
torch.save(merged_tensor, "output/merged_embeddings.pt")

print(f"✅ Merged {len(pt_files)} .pt files into output/merged_embeddings.pt")
print(f"🧠 Final shape: {merged_tensor.shape}")
