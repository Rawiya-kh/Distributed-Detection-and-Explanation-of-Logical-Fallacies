"""
utils/logger.py
---------------
Centralised logging setup for the Fallacy Detection project.

Every script calls `get_logger(__name__)` to get a consistent,
colour-coded, timestamped logger that writes to both STDOUT and a
rotating file in outputs/.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_dir: str = "outputs", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger with:
        - StreamHandler  → coloured console output
        - RotatingFileHandler → persistent log in outputs/

    Parameters
    ----------
    name     : module/script name, typically __name__
    log_dir  : directory where fallacy_project.log is saved
    level    : default logging level

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -- Console handler --
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # -- File handler --
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        filename=os.path.join(log_dir, "fallacy_project.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Prevent propagation to root logger (avoids duplicate lines)
    logger.propagate = False
    return logger
