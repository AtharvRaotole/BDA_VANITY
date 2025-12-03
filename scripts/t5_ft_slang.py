"""
T5 Slang Definition Fine-tuning Script
======================================
Fine-tunes a T5-large model to translate slang words into their definitions
using the Urban Dictionary dataset.

This model learns the relationship between informal language/slang terms
and their meanings, which can be applied to understanding vanity plates.

Dataset: Urban Dictionary word definitions (urbandict-word-defs.csv)
Model: T5-large from Hugging Face Transformers

Features:
- Train/Validation/Test split with reproducible seeding
- Early stopping to prevent overfitting
- Model checkpointing for best validation loss
- Comprehensive logging of training progress
- CSV export of test predictions

Usage:
    python t5_ft_slang.py

Author: AMS 560
Last Updated: December 2025
"""

import pandas as pd
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import project configuration
try:
    from config import (
        DEVICE, DATA_DIR, LOGS_DIR, 
        MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR,
        SLANG_WORD_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT
    )
except ImportError:
    # Fallback for standalone execution
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(__file__).parent.parent / "data"
    LOGS_DIR = Path(__file__).parent.parent / "logs"
    MODEL_CHECKPOINTS_DIR = Path(__file__).parent.parent / "model_checkpoints"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    SLANG_WORD_PROMPT = "Translate this word: {word}"
    LOG_FORMAT = "%(asctime)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

MODEL_NAME = "t5-large"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-4
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 128
SEED = 42

# Data limiting (set to None for all data)
MAX_SAMPLES = 50000

# Early stopping configuration
PATIENCE = 3
MIN_DELTA = 1e-4


# =============================================================================
# DATASET CLASS
# =============================================================================

class SlangDataset(Dataset):
    """
    PyTorch Dataset for slang word definitions.
    
    Tokenizes slang words and their corresponding definitions
    for T5 seq-to-seq training.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing word and definition columns.
        tokenizer (T5Tokenizer): Tokenizer for text encoding.
        max_input_len (int): Maximum length for input sequences.
        max_target_len (int): Maximum length for target sequences.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer: T5Tokenizer, 
        max_input_len: int, 
        max_target_len: int
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: DataFrame with 'word' and 'definition' columns.
            tokenizer: Pre-trained T5 tokenizer.
            max_input_len: Maximum input sequence length.
            max_target_len: Maximum target sequence length.
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        word = str(self.data.iloc[idx]["word"])
        definition = str(self.data.iloc[idx]["definition"])

        # Format input with prompt template
        input_text = SLANG_WORD_PROMPT.format(word=word)
        target_text = definition

        # Tokenize input sequence
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target sequence
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data(filepath: Path, max_samples: int = None) -> pd.DataFrame:
    """
    Load and preprocess the Urban Dictionary dataset.
    
    Args:
        filepath: Path to the CSV file.
        max_samples: Maximum number of samples to use (None for all).
        
    Returns:
        Preprocessed DataFrame with word and definition columns.
    """
    data = pd.read_csv(filepath, on_bad_lines='skip')
    data = data.reset_index(drop=True)
    
    if max_samples is not None:
        data = data.head(max_samples)
    
    data["word"] = data["word"].fillna("").astype(str)
    data["definition"] = data["definition"].fillna("").astype(str)
    
    return data


def create_data_splits(
    data: pd.DataFrame, 
    tokenizer: T5Tokenizer,
    test_size: float = 0.2,
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Create train, validation, and test data splits with DataLoaders.
    
    Args:
        data: Full dataset DataFrame.
        tokenizer: T5 tokenizer for encoding.
        test_size: Proportion of data for validation + test.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, test_data).
    """
    # Split into train and temp (val + test)
    train_data, temp_data = train_test_split(
        data, test_size=test_size, random_state=seed
    )
    
    # Split temp into validation and test
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=seed
    )

    # Create Dataset instances
    train_dataset = SlangDataset(
        train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    val_dataset = SlangDataset(
        val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    test_dataset = SlangDataset(
        test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_data


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: T5ForConditionalGeneration,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: T5 model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for parameter updates.
        device: Device to run training on.
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(train_loader)


def validate(
    model: T5ForConditionalGeneration,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Validate the model on the validation set.
    
    Args:
        model: T5 model to validate.
        val_loader: DataLoader for validation data.
        device: Device to run validation on.
        
    Returns:
        Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            val_loss += outputs.loss.item()

    return val_loss / len(val_loader)


def generate_predictions(
    model: T5ForConditionalGeneration,
    test_loader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device
) -> Tuple[List[str], List[str]]:
    """
    Generate predictions on the test set.
    
    Args:
        model: Trained T5 model.
        test_loader: DataLoader for test data.
        tokenizer: Tokenizer for decoding.
        device: Device to run inference on.
        
    Returns:
        Tuple of (predictions, references).
    """
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_length=MAX_TARGET_LENGTH
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)

    return predictions, references


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    """Main training function."""
    
    # Generate timestamp for this run
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    log_filename = LOGS_DIR / f"training_logs_t5_slang_{time_now}.txt"
    logging.basicConfig(
        filename=str(log_filename),
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    logging.info("=" * 60)
    logging.info("T5 Slang Definition Fine-tuning")
    logging.info("=" * 60)
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Max Epochs: {EPOCHS}")
    logging.info(f"Max Samples: {MAX_SAMPLES}")
    
    # Load data
    logging.info("Loading dataset...")
    data = load_data(DATA_DIR / "urbandict-word-defs.csv", max_samples=MAX_SAMPLES)
    logging.info(f"Loaded {len(data)} samples")
    
    # Initialize tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    
    # Create data splits
    logging.info("Creating data splits...")
    train_loader, val_loader, test_loader, test_data = create_data_splits(
        data, tokenizer
    )
    logging.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Initialize model
    logging.info("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = MODEL_CHECKPOINTS_DIR / f"best_t5_ft_slang_model_{time_now}"
    
    # Training loop with early stopping
    logging.info("Starting training...")
    for epoch in range(EPOCHS):
        logging.info(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train
        avg_train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        logging.info(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validate
        avg_val_loss = validate(model, val_loader, DEVICE)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            model.save_pretrained(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            logging.info(f"✓ New best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= PATIENCE:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    logging.info(f"\nTraining complete. Best model saved to: {best_model_path}")
    
    # Generate test predictions
    logging.info("\nGenerating test predictions...")
    predictions, references = generate_predictions(
        model, test_loader, tokenizer, DEVICE
    )
    
    # Get slang words for results
    test_words = [test_data.iloc[idx]["word"] for idx in range(len(test_loader.dataset))]
    
    # Save results
    results_df = pd.DataFrame({
        "Word": test_words,
        "Predicted": predictions,
        "Reference": references,
    })
    
    output_path = OUTPUTS_DIR / f"slang_t5_ft_predictions_{time_now}.csv"
    results_df.to_csv(str(output_path), index=False)
    logging.info(f"Results saved to: {output_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Training completed successfully!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
