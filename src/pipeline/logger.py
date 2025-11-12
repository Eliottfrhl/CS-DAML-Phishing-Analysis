# src/pipeline/logger.py
from __future__ import annotations
import logging, os, sys, time, random
from contextlib import contextmanager
from typing import Iterable, TypeVar, Optional

# tqdm that works both in notebook and CLI
try:
    from tqdm.auto import tqdm  # notebook/CLI-friendly
except Exception:               # pragma: no cover
    from tqdm import tqdm       # fallback

T = TypeVar("T")

# --- Global logger registry to avoid duplicate handlers in notebooks ---
_LOGGERS: dict[str, logging.Logger] = {}

def init_logging(level: int | str = None) -> None:
    """
    Initialize root logging once (idempotent).
    Level may come from env LOG_LEVEL (DEBUG/INFO/WARNING/ERROR) if not provided.
    """
    global _LOGGERS
    if getattr(init_logging, "_initialized", False):
        return

    # Determine level
    lvl = level or os.getenv("LOG_LEVEL", "INFO")
    if isinstance(lvl, str):
        lvl = getattr(logging, lvl.upper(), logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(lvl)
    init_logging._initialized = True  # type: ignore[attr-defined]

def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """
    Get a module-level logger with consistent formatting.
    Idempotent: won’t duplicate handlers across imports/notebook cells.
    """
    init_logging()  # ensure root configured
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    _LOGGERS[name] = logger
    return logger

@contextmanager
def task(logger: logging.Logger, msg: str):
    """
    Context manager to time steps with a neat start/stop pair.
    Usage:
        with task(log, "Training FastText"):
            ...
    """
    logger.info(f"▶ {msg} ...")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"✓ {msg} done in {dt:.1f}s")

def pbar(iterable: Iterable[T], desc: str = "", total: Optional[int] = None) -> Iterable[T]:
    """
    Thin wrapper on tqdm.auto.tqdm for consistent style across scripts.
    Example:
        for x in pbar(items, desc="Tokenizing"):
            ...
    """
    return tqdm(iterable, desc=desc, total=total, leave=False)

def set_seed(seed: int = 42) -> None:
    """
    Lightweight, global seed helper so every script can call it.
    (Add numpy/torch seeds here if/when needed.)
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
