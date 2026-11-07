# Distributed Logical Fallacy Detection System

A research-grade NLP pipeline that detects and explains logical fallacies in text, combining transformer-based classification, dense retrieval (RAG-style), and parallel/distributed computing techniques.

---

## Project Overview

| Component | Model | Task |
|-----------|-------|------|
| Fallacy Classifier | DeBERTa-v3-xsmall | 10-class text classification |
| Explanation Generator | T5-small | Seq2seq natural language explanation |
| Retrieval Index | FAISS IVFFlat + MiniLM | Dense passage retrieval |

---

## Architecture

```
Raw CSVs
  │
  ▼
01_download_datasets.py   → data/processed/ (train/val/test splits)
  │
  ▼
02_preprocess.py          → NER normalisation (parallel, multiprocessing)
  │
  ▼
03_build_retrieval_corpus → corpus embeddings (sentence-transformers)
  │
  ▼
04_build_faiss_index      → FAISS IVFFlat index (ANN search)
  │
  ├──▶ 05_train_classifier.py  → DeBERTa model (models/deberta_classifier/)
  │
  ├──▶ 06_prepare_generator_data.py  → JSONL training data
  │
  └──▶ 07_train_generator.py   → T5-small model (models/generator/)
```

---

## NLP Concepts Used

- **Transformer fine-tuning**: DeBERTa for sequence classification
- **Disentangled attention**: DeBERTa's two-stream attention mechanism for better token-position modelling
- **Seq2Seq generation**: T5 encoder-decoder with teacher forcing
- **Dense retrieval**: sentence-transformer embeddings + FAISS ANN
- **NER normalisation**: replacing named entities with type placeholders to reduce data sparsity
- **Dynamic padding**: per-batch padding for seq2seq efficiency
- **Macro F1**: balanced metric for imbalanced multi-class classification

---

## Parallel & Distributed Computing Concepts

| Technique | Where Used | Benefit |
|-----------|-----------|---------|
| `ProcessPoolExecutor` | Script 02 | True CPU parallelism (bypasses GIL) for NER preprocessing |
| `DataLoader(num_workers=4)` | Scripts 05, 07 | Async batch prefetching during GPU training |
| `pin_memory=True` | Scripts 05, 07 | Faster CPU→GPU transfers |
| Gradient Accumulation | Scripts 05, 07 | Simulate large batch on small GPU/CPU |
| `torch.nn.DataParallel` | Scripts 05, 07 | Split batches across multiple GPUs |
| `torch.cuda.amp` | Scripts 05, 07 | FP16 mixed precision (2× GPU throughput) |
| Chunked encoding | Script 03 | Memory-efficient passage embedding |
| FAISS IVF + OpenMP | Script 04 | Parallel ANN index construction |
| `datasets.map(num_proc=N)` | Scripts 02 | HuggingFace parallel dataset processing |

---

## Folder Structure

```
fallacy_project/
├── configs/        config.yaml — all hyperparameters
├── data/
│   ├── raw/        original CSVs
│   ├── processed/  train/val/test splits + label_map.json
│   ├── retrieval_corpus/  Kialo passages + embeddings
│   └── generator_data/    T5 JSONL training pairs
├── index/faiss/    FAISS .index file + retrieval map
├── models/
│   ├── deberta_classifier/  checkpoints + final model
│   └── generator/           checkpoints + final model
├── outputs/        logs, metrics, plots
├── scripts/        07 numbered pipeline scripts
└── utils/          logger, seed, device, checkpointing, metrics
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Copy your uploaded CSVs to data/raw/

```bash
cp final_cleaned_dataset.csv data/raw/
cp kialo_flat_clean.csv      data/raw/
cp snli_clean.csv            data/raw/
```

### 3. Configure (optional)

Edit `configs/config.yaml` to change model names, batch sizes, epochs, etc.

---

## Running the Pipeline

Run scripts in order:

```bash
python scripts/01_download_datasets.py
python scripts/02_preprocess.py
python scripts/03_build_retrieval_corpus.py
python scripts/04_build_faiss_index.py
python scripts/05_train_classifier.py
python scripts/06_prepare_generator_data.py
python scripts/07_train_generator.py
```

Each script is independently re-runnable and skips steps already completed.

---

## CPU Optimisation Strategies

1. **OMP_NUM_THREADS**: Set in `.env` to control OpenMP thread count for FAISS/MKL.
2. **Chunked preprocessing**: Process in `chunk_size=200` row batches to avoid peak RAM.
3. **Gradient accumulation**: `batch_size=4, accum=8` → effective batch 32, RAM-friendly.
4. **FP32 precision**: CPU runs in FP32 (FP16 not beneficial on CPU).
5. **DataLoader workers**: Set to 0 on Windows; 4 on Linux/Mac.
6. **Model fallbacks**: xsmall DeBERTa if base is too large; t5-small is fast on CPU.

---

## Free GPU Usage (Colab / Kaggle)

### Google Colab

```python
# Mount drive or clone project
!git clone <your_repo> fallacy_project
%cd fallacy_project
!pip install -r requirements.txt -q
!python -m spacy download en_core_web_sm -q

# Upload CSVs via Files panel or from Drive
!python scripts/01_download_datasets.py
# ... run remaining scripts
```

### Kaggle

1. Upload project as a Dataset.
2. Add CSVs as a Dataset input.
3. Enable GPU accelerator in Settings.
4. Set `KAGGLE_DATA_DIR` and update paths in config.yaml.

### HuggingFace Spaces / Notebooks

Use a T4 GPU notebook.  Mixed precision is auto-enabled when CUDA is detected.

---

## Expected Training Times

| Script | CPU (8-core) | Colab T4 GPU |
|--------|-------------|--------------|
| 01 — Dataset prep | < 1 min | < 1 min |
| 02 — Preprocessing | 5–10 min | 3–5 min |
| 03 — Corpus embeddings | 15–30 min | 3–5 min |
| 04 — FAISS index | 2–5 min | 1–2 min |
| 05 — DeBERTa (5 epochs) | 4–8 hours | 30–60 min |
| 06 — Generator data prep | < 1 min | < 1 min |
| 07 — T5 (10 epochs) | 6–12 hours | 45–90 min |

---

## Memory Optimisation Tips

- Reduce `batch_size` to 2 (classifier) or 1 (generator) on <8 GB RAM.
- Increase `gradient_accumulation_steps` to compensate.
- Use `distilbert-base-uncased` instead of DeBERTa if RAM is below 6 GB.
- Disable `pin_memory` on CPU-only machines (no benefit, saves RAM).
- Process FAISS corpus in chunks (already implemented in script 03).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Lower `batch_size`; increase `gradient_accumulation_steps` |
| `ModuleNotFoundError: spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| `FileNotFoundError: label_map.json` | Run script 01 first |
| FAISS training fails (too few vectors) | Script 04 auto-falls back to FlatIP exact search |
| Windows multiprocessing issues | Set `num_workers: 0` in config.yaml |
| T5 tokeniser `as_target_tokenizer` warning | Update to `transformers>=4.35` |

---

## Reproducibility

All scripts call `set_seed(42)` before any stochastic operation.
The label map is saved alongside the model for deterministic label decoding.
Training history (loss, metrics, epoch) is saved as JSON for analysis.
