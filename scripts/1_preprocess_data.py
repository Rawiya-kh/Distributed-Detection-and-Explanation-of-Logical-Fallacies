# ============================================================================
# COMPLETE WORKING PARALLEL PREPROCESSING - FULLY FIXED
# Run this ENTIRE cell
# ============================================================================

import subprocess
import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PARALLEL PREPROCESSING PIPELINE (FULLY FIXED)")
print("="*60)

# ============================================================================
# STEP 1: INSTALL MISSING DEPENDENCIES
# ============================================================================

def install_if_missing(package, import_name=None):
    import_name = import_name or package.split('>')[0].split('=')[0]
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])
        return False

# Core packages
packages_needed = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "spacy",
    "tqdm",
    "torch",
]

print("\n🔧 Checking dependencies...")
for pkg in packages_needed:
    install_if_missing(pkg)

# Install FAISS CPU
install_if_missing("faiss-cpu", "faiss")

# Download spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)

print("\n✅ All dependencies ready!")

# ============================================================================
# STEP 2: IMPORTS
# ============================================================================

import json
import os
import re
import unicodedata
from collections import Counter
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm
import torch

# Parallel imports
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# STEP 3: CONFIGURATION
# ============================================================================

BASE_DIR = "/content"
DATA_RAW = f"{BASE_DIR}/data/raw"
DATA_PROCESSED = f"{BASE_DIR}/data/processed"
DATA_RETRIEVAL = f"{BASE_DIR}/data/retrieval"

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_RETRIEVAL, exist_ok=True)

NUM_CPUS = cpu_count()
NUM_WORKERS = min(NUM_CPUS, 4)

print(f"\n📊 System Info:")
print(f"   CPU Cores: {NUM_CPUS}")
print(f"   Workers: {NUM_WORKERS}")
print(f"   GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# STEP 4: CONSTANTS (DEFINED ONCE AT TOP LEVEL)
# ============================================================================

ENTITY_MAP = {
    "PERSON": "[PERSON]",
    "ORG": "[ORG]",
    "GPE": "[GPE]",
    "LOC": "[LOC]",
    "NORP": "[NORP]",
    "DATE": "[DATE]",
    "TIME": "[TIME]",
    "MONEY": "[MONEY]",
    "PERCENT": "[PERCENT]",
    "EVENT": "[EVENT]",
}

LABEL_MAPPING = {
    'intentional': 'intentional',
    'fallacy of credibility': 'fallacy_of_credibility',
    'false dilemma': 'false_dilemma',
    'equivocation': 'equivocation',
    'faulty generalization': 'faulty_generalization',
    'ad populum': 'appeal_to_popularity',
    'ad hominem': 'ad_hominem',
    'false causality': 'false_cause',
    'fallacy of logic': 'logical_fallacy',
    'fallacy of relevance': 'relevance_fallacy',
    'appeal to emotion': 'appeal_to_emotion',
    'fallacy of extension': 'straw_man',
    'circular reasoning': 'circular_reasoning',
    'appeal to popularity': 'appeal_to_popularity',
    'false cause': 'false_cause',
    'hasty generalization': 'faulty_generalization',
    'logical fallacy': 'logical_fallacy',
    'relevance fallacy': 'relevance_fallacy',
    'straw man': 'straw_man',
}

# ============================================================================
# STEP 5: TEXT PROCESSING FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFKC', text)
    return text.lower()

def replace_entities_in_text(text: str, nlp) -> str:
    """Replace named entities with placeholders using provided nlp model"""
    if not text or len(text) < 10:
        return text
    
    doc = nlp(text[:3000])
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    for ent in entities:
        placeholder = ENTITY_MAP.get(ent.label_)
        if placeholder:
            text = text[:ent.start_char] + placeholder + text[ent.end_char:]
    
    return text.lower()

# Global variable for spaCy model (will be set per process)
_worker_nlp = None

def init_worker():
    """Initialize spaCy model in each worker process"""
    global _worker_nlp
    import spacy
    _worker_nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger'])

def replace_entities_worker(text: str) -> str:
    """Process single text for NER replacement (top-level function for pickling)"""
    global _worker_nlp
    if not text or len(text) < 10:
        return text
    
    # Clean text first
    clean_text = normalize_text(text)
    
    doc = _worker_nlp(clean_text[:3000])
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    for ent in entities:
        placeholder = ENTITY_MAP.get(ent.label_)
        if placeholder:
            clean_text = clean_text[:ent.start_char] + placeholder + clean_text[ent.end_char:]
    
    return clean_text

def process_text_parallel(texts: List[str]) -> List[str]:
    """Process texts in parallel using multiprocessing"""
    
    if len(texts) < 100:
        # Small dataset: sequential is faster
        print("   Using sequential processing (small dataset)...")
        nlp = spacy.load("en_core_web_sm")
        results = []
        for text in tqdm(texts, desc="NER normalization"):
            clean_text = normalize_text(text)
            clean_text = replace_entities_in_text(clean_text, nlp)
            results.append(clean_text)
        return results
    
    # Use multiprocessing for larger datasets
    print(f"   Using multiprocessing with {NUM_WORKERS} workers...")
    
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(replace_entities_worker, texts, chunksize=50),
            total=len(texts),
            desc="Parallel NER"
        ))
    
    return results

# ============================================================================
# STEP 6: LOAD DATASETS
# ============================================================================

print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

def load_dataset(file_path, label_col, text_col):
    """Load and preprocess a dataset"""
    if not os.path.exists(file_path):
        print(f"   ⚠️ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"   ✅ Loaded: {os.path.basename(file_path)} - {df.shape}")
    
    df['label'] = df[label_col].map(LABEL_MAPPING)
    df = df[df['label'].notna()]
    df['text'] = df[text_col].fillna('').astype(str)
    
    return df[['text', 'label']]

# Load datasets
datasets = {}
climate_path = f"{DATA_RAW}/climate_train.csv"
edu_path = f"{DATA_RAW}/edu_train.csv"

if os.path.exists(climate_path):
    datasets['climate'] = load_dataset(climate_path, "logical_fallacies", "source_article")
else:
    print(f"   ❌ Missing: climate_train.csv")

if os.path.exists(edu_path):
    datasets['edu'] = load_dataset(edu_path, "updated_label", "masked_articles")
else:
    print(f"   ❌ Missing: edu_train.csv")

if not datasets:
    print("\n❌ No datasets found! Please upload files to /content/data/raw/")
    print("   Required: climate_train.csv, edu_train.csv")
    exit()

# Combine datasets
combined = pd.concat(datasets.values(), ignore_index=True)
print(f"\n📊 Combined dataset: {len(combined)} examples")

# Clean
combined = combined.drop_duplicates(subset=['text'])
combined = combined[combined['text'].str.len() > 50]
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   After cleaning: {len(combined)} examples")

# Label distribution
print("\n📊 Label distribution:")
label_counts = Counter(combined['label'])
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"   {label}: {count}")

# ============================================================================
# STEP 7: APPLY PARALLEL NER NORMALIZATION
# ============================================================================

print("\n" + "="*60)
print("APPLYING PARALLEL NER NORMALIZATION")
print("="*60)

texts = combined['text'].tolist()
normalized_texts = process_text_parallel(texts)
combined['text'] = normalized_texts

print(f"   ✅ Normalized {len(normalized_texts)} texts")

# ============================================================================
# STEP 8: CREATE TRAIN/VAL/TEST SPLITS
# ============================================================================

print("\n" + "="*60)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("="*60)

unique_labels = sorted(combined['label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
print(f"Total classes: {len(label_map)}")

combined['label_id'] = combined['label'].map(label_map)

# First split: train_val (90%) and test (10%)
train_val, test = train_test_split(
    combined, test_size=0.1, random_state=42, stratify=combined['label_id']
)

# Second split: train (90% of train_val) and val (10% of train_val)
train, val = train_test_split(
    train_val, test_size=0.111, random_state=42, stratify=train_val['label_id']
)

print(f"\nSplit sizes:")
print(f"   Train: {len(train)}")
print(f"   Val: {len(val)}")
print(f"   Test: {len(test)}")

# Save splits
train.to_csv(f"{DATA_PROCESSED}/train.csv", index=False)
val.to_csv(f"{DATA_PROCESSED}/val.csv", index=False)
test.to_csv(f"{DATA_PROCESSED}/test.csv", index=False)

with open(f"{DATA_PROCESSED}/label_map.json", 'w') as f:
    json.dump(label_map, f, indent=2)

print("\n✅ Splits saved to /content/data/processed/")

# ============================================================================
# STEP 9: CREATE RETRIEVAL CORPUS
# ============================================================================

print("\n" + "="*60)
print("CREATING RETRIEVAL CORPUS")
print("="*60)

kialo_path = f"{DATA_RAW}/kialo_flat_clean.csv"
corpus = []

if os.path.exists(kialo_path):
    kialo_df = pd.read_csv(kialo_path)
    print(f"   Loaded Kialo: {len(kialo_df)} arguments")
    
    # Load spaCy for Kialo processing (separate instance)
    nlp_kialo = spacy.load("en_core_web_sm", disable=['parser', 'tagger'])
    
    for idx, row in tqdm(kialo_df.iterrows(), total=len(kialo_df), desc="Processing Kialo"):
        # Get argument text
        text = str(row.get('argument', '')) if pd.notna(row.get('argument', '')) else ""
        
        if len(text) > 50:
            # Normalize
            clean_text = normalize_text(text)
            
            # Replace entities using the globally defined ENTITY_MAP
            doc = nlp_kialo(clean_text[:3000])
            entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
            
            for ent in entities:
                placeholder = ENTITY_MAP.get(ent.label_)  # ENTITY_MAP is now defined globally
                if placeholder:
                    clean_text = clean_text[:ent.start_char] + placeholder + clean_text[ent.end_char:]
            
            corpus.append({
                "id": idx,
                "text": text[:500],
                "text_normalized": clean_text,
                "question": str(row.get('question', ''))[:200] if pd.notna(row.get('question', '')) else "",
                "type": str(row.get('type', '')) if pd.notna(row.get('type', '')) else ""
            })
    
    # Save corpus
    corpus_path = f"{DATA_RETRIEVAL}/retrieval_corpus.jsonl"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Created corpus with {len(corpus)} passages")
    print(f"   📁 Saved to: {corpus_path}")
else:
    print(f"   ⚠️ Kialo data not found at {kialo_path}")
    print(f"   Skipping retrieval corpus creation")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("✅ PARALLEL PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n📊 Final Dataset Summary:")
print(f"   Training examples: {len(train)}")
print(f"   Validation examples: {len(val)}")
print(f"   Test examples: {len(test)}")
print(f"   Fallacy types: {len(label_map)}")
print(f"   Retrieval passages: {len(corpus)}")
print(f"\n📁 Output directory: {DATA_PROCESSED}")
print(f"\n📁 Retrieval corpus: {DATA_RETRIEVAL}")
print(f"\n🚀 Next steps:")
print(f"   1. Build FAISS index: python 2_build_faiss_index.py")
print(f"   2. Train classifier: python 3_train_classifier.py")