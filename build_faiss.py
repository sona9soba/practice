import faiss
import torch

# Load your merged tensor
emb = torch.load("./mce_leadlike.pt").numpy().astype("float32")

# Create HNSW index (M=32 is a good default)
index = faiss.IndexHNSWFlat(emb.shape[1], 32)
index.hnsw.efConstruction = 200
index.add(emb)

# Save for later
faiss.write_index(index, "faiss_hnsw.index")
