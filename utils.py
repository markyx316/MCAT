"""
utils.py — Shared utilities: logging, timing, reproducibility.
"""

import logging
import time
import random
import functools
import numpy as np

from config import LOG_LEVEL, RANDOM_SEED


def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def timer(func):
    """Decorator that logs execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed < 60:
            logger.info(f"{func.__name__} completed in {elapsed:.1f}s")
        else:
            logger.info(f"{func.__name__} completed in {elapsed/60:.1f}min")
        return result
    return wrapper


def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
