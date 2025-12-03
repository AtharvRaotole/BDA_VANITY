"""
Configuration Management Module
===============================
Centralized configuration for the Vanity Plate Interpretation project.

This module provides all hyperparameters, paths, and settings used across
the training and inference scripts. Modify values here to change behavior
across all scripts.

Author: AMS 560
Last Updated: December 2025
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIRS = PROJECT_ROOT / "data_dirs"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "model_checkpoints"
SLURM_LOGS_DIR = PROJECT_ROOT / "slurm_logs"

# Ensure directories exist
for directory in [LOGS_DIR, OUTPUTS_DIR, MODEL_CHECKPOINTS_DIR, SLURM_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET PATHS
# =============================================================================

# California vanity plate datasets
CALI_DATASET = DATA_DIR / "cali.csv"
CALI_V2_DATASET = DATA_DIR / "cali_v2.csv"
CALI_V2_LLAMA_RC_DATASET = DATA_DIR / "cali_v2_llama_rc.csv"
CALI_SLANG_ABB_DATASET = DATA_DIR / "cali_slang_abb.csv"

# Slang and abbreviation datasets
SLANG_ABB_DATASET = DATA_DIR / "slang_abb.csv"
URBAN_DICT_DATASET = DATA_DIR / "urbandict-word-defs.csv"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class T5Config:
    """Configuration for T5 model training and inference."""
    
    model_name: str = "t5-large"
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 5e-5
    max_input_length: int = 32
    max_target_length: int = 128
    
    # Early stopping parameters
    patience: int = 5
    min_delta: float = 1e-4
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class BERTConfig:
    """Configuration for BERT model training and inference."""
    
    model_name: str = "bert-base-uncased"
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 5e-5
    max_length: int = 128
    
    # Early stopping parameters
    patience: int = 2
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class SlangT5Config:
    """Configuration for T5 slang fine-tuning."""
    
    model_name: str = "t5-large"
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 2e-4
    max_input_length: int = 32
    max_target_length: int = 128
    
    # Early stopping parameters
    patience: int = 3
    min_delta: float = 1e-4
    
    # Number of samples to use (None for all)
    max_samples: Optional[int] = 50000
    
    # Random seed for reproducibility
    seed: int = 42


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device() -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


DEVICE = get_device()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Template for vanity plate translation
VANITY_PLATE_PROMPT = "Translate vanity plate: {plate}"

# Template for slang word translation
SLANG_WORD_PROMPT = "Translate this word: {word}"


# =============================================================================
# DATA SPLIT RATIOS
# =============================================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1  # Note: 0.2 total for val+test, then split 50/50


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        str: Timestamp in YYYYMMDD_HHMMSS format.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_save_path(model_type: str, suffix: str = "") -> Path:
    """
    Generate a standardized model save path.
    
    Args:
        model_type: Type of model (e.g., 'vanity_plate', 't5_slang')
        suffix: Optional suffix for the path
        
    Returns:
        Path: Full path for saving the model.
    """
    timestamp = get_timestamp()
    name = f"best_{model_type}_model_{timestamp}"
    if suffix:
        name = f"{name}_{suffix}"
    return MODEL_CHECKPOINTS_DIR / name


def get_log_path(prefix: str) -> Path:
    """
    Generate a standardized log file path.
    
    Args:
        prefix: Prefix for the log file name.
        
    Returns:
        Path: Full path for the log file.
    """
    timestamp = get_timestamp()
    return LOGS_DIR / f"training_logs_{prefix}_{timestamp}.txt"


def get_output_path(prefix: str, extension: str = "csv") -> Path:
    """
    Generate a standardized output file path.
    
    Args:
        prefix: Prefix for the output file name.
        extension: File extension (default: csv).
        
    Returns:
        Path: Full path for the output file.
    """
    timestamp = get_timestamp()
    return OUTPUTS_DIR / f"{prefix}_{timestamp}.{extension}"

