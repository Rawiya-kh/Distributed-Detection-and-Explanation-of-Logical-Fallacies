# ============================================================================
# FILE: 6_complete_pipeline_parallel.py
# CLEAN PARALLEL INFERENCE PIPELINE - No Web Dependencies, Fixed Asyncio
# ============================================================================

import json
import os
import torch
import numpy as np
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CPUS = mp.cpu_count()
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE = 32
MAX_WORKERS = min(NUM_CPUS, 8)

print("="*60)
print("PARALLEL INFERENCE PIPELINE (CLEAN VERSION)")
print("="*60)
print(f"CPU Cores: {NUM_CPUS}")
print(f"GPUs: {NUM_GPUS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Workers: {MAX_WORKERS}")

# ============================================================================
# CLASS 1: THREADED BATCH PROCESSOR (No asyncio)
# ============================================================================

class ThreadedBatchProcessor:
    """Process inference requests in batches using threading (no asyncio)"""
    
    def __init__(self, model, tokenizer, device, id_to_label, batch_size=32, max_wait_time=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.id_to_label = id_to_label
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        self.queue = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        self.results = {}
        
        # Start background thread
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
    
    def add_request(self, text: str) -> Dict:
        """Add single request and wait for result"""
        request_id = id(text)
        future = {'ready': False, 'result': None}
        
        with self.condition:
            self.queue.append((text, request_id, future))
            self.condition.notify()
        
        # Wait for result
        while not future['ready']:
            time.sleep(0.001)
        
        return future['result']
    
    def add_batch(self, texts: List[str]) -> List[Dict]:
        """Add multiple requests and wait for all results"""
        return [self.add_request(text) for text in texts]
    
    def _process_loop(self):
        """Background thread processing loop"""
        while self.running:
            batch = []
            
            with self.condition:
                if not self.queue:
                    self.condition.wait(timeout=self.max_wait_time)
                    continue
                
                # Get up to batch_size items
                for _ in range(min(self.batch_size, len(self.queue))):
                    batch.append(self.queue.popleft())
            
            if batch:
                self._process_batch(batch)
    
    def _process_batch(self, batch):
        """Process a batch of requests"""
        texts = [text for text, _, _ in batch]
        request_ids = [rid for _, rid, _ in batch]
        futures = [future for _, _, future in batch]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # GPU Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_ids = outputs.logits.argmax(dim=-1)
        
        # Set results
        for i, (text, rid, future) in enumerate(batch):
            future['result'] = {
                'text': text[:100],
                'fallacy': self.id_to_label[pred_ids[i].item()],
                'confidence': probs[i][pred_ids[i]].item()
            }
            future['ready'] = True
    
    def shutdown(self):
        """Stop the background thread"""
        self.running = False
        with self.condition:
            self.condition.notify_all()

# ============================================================================
# CLASS 2: PARALLEL BATCH PROCESSOR (Multi-threading)
# ============================================================================

class ParallelBatchProcessor:
    """Process batches in parallel using ThreadPoolExecutor"""
    
    def __init__(self, model, tokenizer, device, id_to_label):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.id_to_label = id_to_label
    
    def predict_batch_parallel(self, texts: List[str], num_workers: int = None) -> List[Dict]:
        """Process batch using multiple threads"""
        if num_workers is None:
            num_workers = min(MAX_WORKERS, len(texts))
        
        def predict_one(text):
            return self._predict_single(text)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(predict_one, texts),
                total=len(texts),
                desc="Parallel inference"
            ))
        
        return results
    
    def predict_batch_gpu(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """GPU-optimized batch prediction"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_ids = outputs.logits.argmax(dim=-1)
            
            for j, text in enumerate(batch_texts):
                results.append({
                    'text': text[:100],
                    'fallacy': self.id_to_label[pred_ids[j].item()],
                    'confidence': probs[j][pred_ids[j]].item()
                })
        
        return results
    
    def _predict_single(self, text: str) -> Dict:
        """Single text prediction"""
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_id = outputs.logits.argmax().item()
        
        return {
            'text': text[:100],
            'fallacy': self.id_to_label[pred_id],
            'confidence': probs[0][pred_id].item()
        }

# ============================================================================
# CLASS 3: FAISS RETRIEVAL (Optional)
# ============================================================================

class FaissRetriever:
    """FAISS-based retrieval for similar examples"""
    
    def __init__(self, index_path="/content/index/faiss/corpus.index", 
                 metadata_path="/content/index/faiss/corpus_metadata.json"):
        self.index = None
        self.metadata = None
        self.encoder = None
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✅ Loaded FAISS index with {self.index.ntotal} passages")
            except Exception as e:
                print(f"⚠️ Could not load FAISS index: {e}")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar passages"""
        if self.index is None or self.encoder is None:
            return []
        
        try:
            query_embedding = self.encoder.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    results.append({
                        "text": self.metadata[idx]['text'][:200],
                        "score": float(score)
                    })
            return results
        except Exception as e:
            return []

# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class ParallelPipeline:
    """Complete parallel inference pipeline"""
    
    def __init__(self, model_path="/content/models/classifier", use_retrieval=False):
        print("\n🔧 Initializing Parallel Pipeline...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")
        
        # Load model
        print("   Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load label map
        with open(f"{model_path}/label_map.json", 'r') as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        print(f"   ✅ Loaded {len(self.label_map)} fallacy types")
        
        # Initialize processors
        self.batch_processor = ParallelBatchProcessor(
            self.model, self.tokenizer, self.device, self.id_to_label
        )
        
        self.threaded_processor = ThreadedBatchProcessor(
            self.model, self.tokenizer, self.device, self.id_to_label,
            batch_size=BATCH_SIZE
        )
        
        # Initialize retrieval (optional)
        self.retriever = FaissRetriever() if use_retrieval else None
        
        print("✅ Pipeline ready!\n")
    
    def predict(self, text: str) -> Dict:
        """Predict single text"""
        return self.batch_processor._predict_single(text)
    
    def predict_batch(self, texts: List[str], method: str = 'gpu_batch') -> List[Dict]:
        """Predict batch of texts with different parallel strategies"""
        
        if method == 'gpu_batch':
            return self.batch_processor.predict_batch_gpu(texts)
        elif method == 'thread':
            return self.batch_processor.predict_batch_parallel(texts)
        else:
            return [self.predict(t) for t in texts]
    
    def predict_with_retrieval(self, text: str, k: int = 3) -> Dict:
        """Predict with similar example retrieval"""
        result = self.predict(text)
        
        if self.retriever:
            similar = self.retriever.search(text, k)
            result['similar_examples'] = similar
        
        return result
    
    def benchmark(self, texts: List[str]):
        """Benchmark different parallel methods"""
        print("\n🔬 BENCHMARKING PARALLEL METHODS")
        print("="*40)
        
        methods = [
            ('Sequential', lambda: [self.predict(t) for t in texts]),
            ('GPU Batch', lambda: self.predict_batch(texts, method='gpu_batch')),
            ('Thread Parallel', lambda: self.predict_batch(texts, method='thread')),
        ]
        
        results = {}
        baseline_time = None
        
        for name, method in methods:
            start = time.time()
            method()
            elapsed = time.time() - start
            throughput = len(texts) / elapsed
            
            results[name] = {
                'time': elapsed,
                'throughput': throughput
            }
            
            if baseline_time is None:
                baseline_time = elapsed
            
            print(f"\n📊 {name}:")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {throughput:.1f} texts/sec")
            print(f"   Speedup: {baseline_time/elapsed:.2f}x")
        
        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("FALLACY DETECTION PIPELINE (Parallel Version)")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ParallelPipeline(use_retrieval=False)
    
    # Test arguments
    test_texts = [
        "You can't trust Dr. Smith's research because he drives an SUV!",
        "Everyone is buying this product, so it must be the best.",
        "Since I started wearing this bracelet, I've had good luck.",
        "You're either with us or against us.",
        "I met two rude people from New York, so everyone there is rude.",
        "If we don't pass this law, innocent children will suffer.",
        "My favorite actor says this medicine works.",
    ]
    
    # Benchmark
    pipeline.benchmark(test_texts)
    
    # Interactive demo
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("="*60)
    print("Type 'quit' to exit, 'benchmark' to run benchmark")
    print("-"*40)
    
    while True:
        user_input = input("\n📝 Enter argument: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'benchmark':
            pipeline.benchmark(test_texts)
            continue
        
        if not user_input:
            continue
        
        # Predict
        start = time.time()
        result = pipeline.predict(user_input)
        elapsed = time.time() - start
        
        print(f"\n🎯 Fallacy: {result['fallacy']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"⏱️ Time: {elapsed:.3f}s")
        print("-"*40)

if __name__ == "__main__":
    main()