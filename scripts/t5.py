"""
T5 Vanity Plate Interpretation Training Script
==============================================
Fine-tunes a T5-large model to interpret/translate vanity license plates
into their intended meanings based on DMV reviewer comments.

Dataset: California DMV vanity plate applications (cali.csv)
Model: T5-large from Hugging Face Transformers

Features:
- Train/Validation/Test split with reproducible seeding
- Early stopping to prevent overfitting
- Model checkpointing for best validation loss
- Comprehensive logging of training progress
- CSV export of test predictions

Usage:
    python t5.py

Author: DSF Project Team
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
        T5Config, DEVICE, DATA_DIR, LOGS_DIR, 
        MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR,
        VANITY_PLATE_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT
    )
except ImportError:
    # Fallback for standalone execution
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(__file__).parent.parent / "data"
    LOGS_DIR = Path(__file__).parent.parent / "logs"
    MODEL_CHECKPOINTS_DIR = Path(__file__).parent.parent / "model_checkpoints"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    VANITY_PLATE_PROMPT = "Translate vanity plate: {plate}"
    LOG_FORMAT = "%(asctime)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

MODEL_NAME = "t5-large"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 100
SEED = 42

# Early stopping configuration
PATIENCE = 5
MIN_DELTA = 1e-4


# =============================================================================
# DATASET CLASS
# =============================================================================

class VanityPlateDataset(Dataset):
    """
    PyTorch Dataset for vanity license plate interpretation.
    
    Tokenizes plate configurations and their corresponding meanings/comments
    for T5 seq-to-seq training.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing plate and comment columns.
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
            dataframe: DataFrame with 'plate' and 'reviewer_comments' columns.
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
        plate = str(self.data.iloc[idx]["plate"])
        comment = str(self.data.iloc[idx]["reviewer_comments"])

        # Format input with prompt template
        input_text = VANITY_PLATE_PROMPT.format(plate=plate)
        target_text = comment

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

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load and preprocess the vanity plate dataset.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Preprocessed DataFrame with plate and reviewer_comments columns.
    """
    data = pd.read_csv(filepath)
    data = data.reset_index(drop=True)
    data["plate"] = data["plate"].fillna("").astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)
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
    train_dataset = VanityPlateDataset(
        train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    val_dataset = VanityPlateDataset(
        val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    test_dataset = VanityPlateDataset(
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
    log_filename = LOGS_DIR / f"training_logs_{time_now}.txt"
    logging.basicConfig(
        filename=str(log_filename),
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    logging.info("=" * 60)
    logging.info("T5 Vanity Plate Interpretation Training")
    logging.info("=" * 60)
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Max Epochs: {EPOCHS}")
    
    # Load data
    logging.info("Loading dataset...")
    data = load_data(DATA_DIR / "cali.csv")
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
    best_model_path = MODEL_CHECKPOINTS_DIR / f"best_vanity_plate_model_{time_now}"
    
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
    
    # Get plate numbers for results
    test_plates = [test_data.iloc[idx]["plate"] for idx in range(len(test_loader.dataset))]
    
    # Save results
    results_df = pd.DataFrame({
        "Plate": test_plates,
        "Predicted": predictions,
        "Reference": references,
    })
    
    output_path = OUTPUTS_DIR / f"vanity_plate_predictions_{time_now}.csv"
    results_df.to_csv(str(output_path), index=False)
    logging.info(f"Results saved to: {output_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Training completed successfully!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
