"""
T5 Vanity Plate Inference Script
================================
Load a trained T5 model and run inference on vanity plate data.

This script loads a pre-trained T5 checkpoint and generates interpretations
for vanity plates in a test dataset.

Usage:
    python t5_inference.py [--model_path PATH] [--input_csv PATH]

Author: DSF Project Team
Last Updated: December 2025
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import project configuration
try:
    from config import (
        DEVICE, DATA_DIR, MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR,
        VANITY_PLATE_PROMPT, get_timestamp
    )
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(__file__).parent.parent / "data"
    MODEL_CHECKPOINTS_DIR = Path(__file__).parent.parent / "model_checkpoints"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    VANITY_PLATE_PROMPT = "Translate vanity plate: {plate}"
    
    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default model timestamp - update this to use your trained model
DEFAULT_MODEL_TIMESTAMP = "20241123_194100"

BATCH_SIZE = 4
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 64
SEED = 42


# =============================================================================
# DATASET CLASS
# =============================================================================

class VanityPlateDataset(Dataset):
    """
    PyTorch Dataset for vanity plate inference.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing plate data.
        tokenizer (T5Tokenizer): Tokenizer for encoding.
        max_input_len (int): Maximum input sequence length.
        max_target_len (int): Maximum target sequence length.
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
        return len(self.data)

    def __getitem__(self, idx: int):
        plate = str(self.data.iloc[idx]["plate"])
        comment = str(self.data.iloc[idx]["reviewer_comments"])

        input_text = VANITY_PLATE_PROMPT.format(plate=plate)
        target_text = comment

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
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
# INFERENCE FUNCTIONS
# =============================================================================

def load_model(model_path: Path) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load a trained T5 model and tokenizer.
    
    Args:
        model_path: Path to the saved model directory.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(str(model_path))
    model = T5ForConditionalGeneration.from_pretrained(str(model_path)).to(DEVICE)
    model.eval()
    return model, tokenizer


def run_inference(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    data_loader: DataLoader
) -> Tuple[List[str], List[str]]:
    """
    Run inference on a dataset.
    
    Args:
        model: Trained T5 model.
        tokenizer: T5 tokenizer.
        data_loader: DataLoader for inference data.
        
    Returns:
        Tuple of (predictions, references).
    """
    predictions = []
    references = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

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


def save_results(
    plates: List[str],
    predictions: List[str],
    references: List[str],
    output_path: Path
) -> None:
    """
    Save inference results to CSV.
    
    Args:
        plates: List of plate configurations.
        predictions: List of model predictions.
        references: List of reference meanings.
        output_path: Path to save the CSV file.
    """
    results_df = pd.DataFrame({
        "Plate": plates,
        "Predicted": predictions,
        "Reference": references,
    })
    results_df.to_csv(str(output_path), index=False)
    print(f"Results saved to: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description="T5 Vanity Plate Inference")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--input_csv", 
        type=str, 
        default=None,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None,
        help="Path to output CSV file"
    )
    args = parser.parse_args()
    
    # Set default paths
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = MODEL_CHECKPOINTS_DIR / f"vanity_plate_model_{DEFAULT_MODEL_TIMESTAMP}"
    
    input_csv = Path(args.input_csv) if args.input_csv else DATA_DIR / "cali.csv"
    
    timestamp = get_timestamp()
    output_csv = (
        Path(args.output_csv) if args.output_csv 
        else OUTPUTS_DIR / f"vanity_plate_predictions_{timestamp}.csv"
    )
    
    # Load data
    print(f"Loading data from: {input_csv}")
    data = pd.read_csv(input_csv)
    data["plate"] = data["plate"].fillna("").astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Split data to get test set (matching training splits)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)
    
    # Create dataset and loader
    test_dataset = VanityPlateDataset(
        test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Run inference
    predictions, references = run_inference(model, tokenizer, test_loader)
    
    # Get plate numbers
    test_plates = [test_data.iloc[idx]["plate"] for idx in range(len(test_dataset))]
    
    # Save results
    save_results(test_plates, predictions, references, output_csv)
    
    # Print some examples
    print("\n" + "=" * 60)
    print("Example Predictions:")
    print("=" * 60)
    for i in range(min(5, len(predictions))):
        print(f"\nPlate: {test_plates[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Reference: {references[i]}")


if __name__ == "__main__":
    main()
