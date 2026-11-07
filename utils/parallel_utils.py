"""
utils/parallel_utils.py
------------------------
CPU parallelism helpers for preprocessing.

WHY PARALLEL PREPROCESSING MATTERS
------------------------------------
Raw text cleaning (HTML removal, spaCy NER, unicode normalisation) is
compute-bound and embarrassingly parallel — each text row is independent.
On an 8-core laptop this can give a 4-6× speedup.

HOW CHUNKING WORKS
-------------------
The dataset is split into N equal chunks (one per CPU core).  Each chunk
is sent to a separate Python process in a Pool.  Because Python's GIL
prevents true threading for CPU work, we use *processes* which each
get their own Python interpreter and memory space.  Results are
concatenated after all processes finish.

HOW CPU UTILISATION IMPROVES
------------------------------
Without chunking: 1 core at 100%, rest idle.
With chunking   : N cores each at ~100% → N× throughput.
"""

import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List

import pandas as pd
from tqdm import tqdm
from utils.logger import get_logger

log = get_logger(__name__)

# ── basic text cleaning (no spaCy — safe to pickle for multiprocessing) ──

def _basic_clean(text: str) -> str:
    """Remove HTML, URLs, excessive whitespace, normalise unicode."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Unicode normalisation (NFKC handles ligatures, full-width chars, etc.)
    text = unicodedata.normalize("NFKC", text)
    return text


def _clean_chunk(chunk: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Worker function: clean a single DataFrame chunk.
    Must be importable at module level for multiprocessing to pickle it.
    """
    chunk = chunk.copy()
    chunk[text_col] = chunk[text_col].apply(_basic_clean)
    return chunk


def parallel_clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    Apply basic text cleaning in parallel across CPU cores.

    Parameters
    ----------
    df         : Input DataFrame.
    text_col   : Column to clean.
    num_workers: Number of parallel processes.

    Returns
    -------
    Cleaned DataFrame (same order as input).
    """
    num_workers = min(num_workers, os.cpu_count() or 1)
    chunks = _split_dataframe(df, num_workers)

    log.info(f"Parallel cleaning: {len(df)} rows → {num_workers} workers × {len(chunks[0])} rows/chunk")

    results = [None] * len(chunks)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_clean_chunk, chunk, text_col): i
            for i, chunk in enumerate(chunks)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning chunks"):
            idx = futures[future]
            results[idx] = future.result()

    return pd.concat(results, ignore_index=True)


def _split_dataframe(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    """Split a DataFrame into n roughly equal chunks."""
    chunk_size = max(1, len(df) // n)
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
