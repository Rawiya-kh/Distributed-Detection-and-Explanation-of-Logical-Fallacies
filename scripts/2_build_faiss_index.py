# ============================================================================
# FIXED FAISS INDEX BUILDER - Works with or without GPU
# ============================================================================

import json
import os
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("="*60)
print("FAISS INDEX BUILDER (CPU/GPU Compatible)")
print("="*60)

# Check FAISS capabilities
HAS_FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
print(f"FAISS GPU Support: {'YES' if HAS_FAISS_GPU else 'NO (using CPU)'}")

class CompatibleFAISSBuilder:
    def __init__(self, model_name='intfloat/e5-large-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        if torch.cuda.is_available():
            self.model.half()
        
        # Store CPU index separately from GPU index
        self.cpu_index = None
    
    def encode_passages(self, passages, batch_size=64):
        texts = [f"passage: {p['text_normalized']}" for p in passages]
        return self.model.encode(texts, batch_size=batch_size, 
                                  show_progress_bar=True, 
                                  normalize_embeddings=True).astype(np.float32)
    
    def build_index(self, embeddings):
        dimension = embeddings.shape[1]
        print(f"Building index (dim={dimension})...")
        
        # Build CPU index first (always keep a copy for saving)
        self.cpu_index = faiss.IndexFlatIP(dimension)
        self.cpu_index.add(embeddings)
        print(f"   Added {self.cpu_index.ntotal} vectors to CPU index")
        
        # Try to create GPU index for faster search
        if HAS_FAISS_GPU and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index)
                print("   ✅ Using GPU index for searches")
                return gpu_index
            except Exception as e:
                print(f"   ⚠️ Could not move to GPU: {e}")
                print("   Using CPU index")
        
        return self.cpu_index
    
    def save_index(self, index, path):
        # Always save the CPU index (not the GPU one)
        if self.cpu_index is not None:
            faiss.write_index(self.cpu_index, path)
            print(f"✅ Index saved to {path}")
        else:
            # Fallback: try to convert GPU to CPU
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, path)
                print(f"✅ Index saved to {path}")
            except:
                print("❌ Could not save index")
                raise

def main():
    corpus_path = "/content/data/retrieval/retrieval_corpus.jsonl"
    index_path = "/content/index/faiss/corpus.index"
    
    os.makedirs("/content/index/faiss", exist_ok=True)
    
    # Load passages
    passages = []
    with open(corpus_path, 'r') as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    print(f"Loaded {len(passages)} passages")
    
    # Build index
    builder = CompatibleFAISSBuilder()
    embeddings = builder.encode_passages(passages)
    print(f"Embeddings shape: {embeddings.shape}")
    
    index = builder.build_index(embeddings)
    builder.save_index(index, index_path)
    
    # Test search
    test_query = "You can't trust his argument because he's a convicted criminal"
    query_emb = builder.model.encode([f"query: {test_query}"], normalize_embeddings=True)
    scores, indices = index.search(query_emb.astype(np.float32), 5)
    
    print("\nTest search results:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(passages):
            print(f"   {i+1}. Score: {score:.4f} - {passages[idx]['text'][:80]}...")

if __name__ == "__main__":
    main()