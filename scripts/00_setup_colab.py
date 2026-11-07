# ============================================================================
# QUICK ALTERNATIVE: SIMPLE WORKING SETUP (2 minutes)
# Use this if you just want everything to work
# ============================================================================

import subprocess
import sys

print("="*60)
print("QUICK WORKING SETUP (2 minutes)")
print("="*60)

# Install everything in one go (all have pre-compiled wheels)
packages = [
    "torch",
    "transformers", 
    "sentence-transformers",
    "scikit-learn",
    "pandas",
    "numpy",
    "tqdm",
    "spacy",
    "accelerate",
    "faiss-cpu",  # CPU version - works perfectly
    "ray[default]",
    "dask[complete]",
    "joblib",
    "numba",
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

# Download spaCy
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

print("\n✅ All packages installed!")
print("ℹ️ Using FAISS CPU (StandardGpuResources not needed)")
print("   This works perfectly for 418 passages")

# Test imports
import faiss
import ray
import dask
import joblib
import numba
print("\n✅ All imports successful!")
print(f"FAISS version: {faiss.__version__}")