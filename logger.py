"""
logger.py — Reusable logging setup for the P2P context engine.

Usage:
    from logger import get_logger
    log = get_logger("schema_agent")
    log.info("Starting...")
"""

import logging
from pathlib import Path
from datetime import datetime


LOG_DIR = Path("logs")


def get_logger(
    name: str,
    log_dir: str | Path = LOG_DIR,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Returns a logger that writes to:
      - console: INFO and above (clean, readable)
      - logs/<name>.log: DEBUG and above (full detail, append mode)

    Calling get_logger() with the same name twice returns the
    same logger instance — safe to call at module level.
    """
    logger = logging.getLogger(name)

    # Guard — don't add handlers twice if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    log_file = log_dir / f"{name}.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Write session separator on first open
    logger.debug("=" * 62)
    logger.debug(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug("=" * 62)

    return logger