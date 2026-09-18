"""
config.py
=============================================================================
Centralized Configuration and Logging Module for Smart Resume Matching System.
Adheres to production standards and VITyarthi AI/ML project guidelines.
=============================================================================
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MLThresholds:
    """Threshold values for candidate evaluation and classification."""
    STRONG_MATCH: float = 0.70       # >= 70% match
    QUALIFIED_MATCH: float = 0.50    # 50% - 69.9% match
    PARTIAL_MATCH: float = 0.30      # 30% - 49.9% match
    # Below 30% is categorized as LOW MATCH / High Gap


@dataclass(frozen=True)
class VectorizerConfig:
    """Hyperparameters for TF-IDF Vectorizer."""
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    MAX_FEATURES: int = 5000
    STOP_WORDS: str = "english"
    SUBLINEAR_TF: bool = True
    NORM: str = "l2"


@dataclass(frozen=True)
class AppConfig:
    """Global system configuration settings."""
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DB_NAME: str = "vityarthi_ml.db"
    LOG_NAME: str = "app.log"
    DB_PATH: str = os.path.join(BASE_DIR, "vityarthi_ml.db")
    LOG_PATH: str = os.path.join(BASE_DIR, "app.log")
    
    ML: MLThresholds = MLThresholds()
    VECTORIZER: VectorizerConfig = VectorizerConfig()


# Singleton application configuration instance
CONFIG = AppConfig()


def setup_logger(name: str = "ResumeMatcher") -> logging.Logger:
    """
    Sets up and returns a production-grade dual-output logger
    (console and app.log) with consistent timestamped formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        log_format = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler (app.log)
        file_handler = logging.FileHandler(CONFIG.LOG_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        # Console (stdout) Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger


# Default logger instance for quick imports
logger = setup_logger()
