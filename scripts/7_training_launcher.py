# ============================================================================
# FILE: 07_training_launcher.py
# Complete training launcher for Colab T4
# ============================================================================

import os
import subprocess
import sys

def run_all_training():
    """Run complete training pipeline"""
    
    print("="*60)
    print("COMPLETE FALLACY DETECTION TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Install dependencies
    print("\n[1/5] Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "torch", "transformers", "datasets", "sentence-transformers",
                   "faiss-gpu", "scikit-learn", "pandas", "numpy", "tqdm",
                   "spacy", "accelerate", "evaluate", "rouge-score", "bert-score"])
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    
    # Step 2: Preprocess data
    print("\n[2/5] Running preprocessing...")
    subprocess.run(["python", "01_preprocess_data.py"])
    
    # Step 3: Build FAISS index
    print("\n[3/5] Building FAISS index...")
    subprocess.run(["python", "02_build_faiss_index.py"])
    
    # Step 4: Train classifier
    print("\n[4/5] Training classifier...")
    subprocess.run(["python", "03_train_classifier.py"])
    
    # Step 5: Train generator
    print("\n[5/5] Training generator...")
    subprocess.run(["python", "04_train_generator.py"])
    
    print("\n" + "="*60)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nTo run the complete pipeline with trained models:")
    print("  python 06_complete_pipeline.py")

if __name__ == "__main__":
    run_all_training()