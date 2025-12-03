"""
T5 Slang Model Inference on California Plates
=============================================
Apply a T5 model trained on slang definitions to interpret
California vanity plates.

This script tests whether understanding slang/abbreviations transfers
to understanding vanity plate meanings.

Usage:
    python t5_slang_inference_on_cali.py

Author: AMS 560
Last Updated: December 2025
"""

import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import project configuration
try:
    from config import (
        DEVICE, DATA_DIR, MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR,
        SLANG_WORD_PROMPT, get_timestamp
    )
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(__file__).parent.parent / "data"
    MODEL_CHECKPOINTS_DIR = Path(__file__).parent.parent / "model_checkpoints"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    SLANG_WORD_PROMPT = "Translate this word: {word}"
    
    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model timestamp - update to use your trained model
MODEL_TIMESTAMP = "20241128_212133"

BATCH_SIZE = 16
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 256
SEED = 42


# =============================================================================
# DATASET CLASS
# =============================================================================

class SlangDataset(Dataset):
    """
    Dataset for applying slang model to plate data.
    
    Uses the slang prompt template with plate configurations.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer: T5Tokenizer, 
        max_input_len: int, 
        max_target_len: int
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        word = str(self.data.iloc[idx]["plate"])
        definition = str(self.data.iloc[idx]["reviewer_comments"])

        input_text = SLANG_WORD_PROMPT.format(word=word)
        target_text = definition

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

def run_inference(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    data_loader: DataLoader
) -> Tuple[List[str], List[str]]:
    """
    Run inference on the dataset.
    
    Args:
        model: Trained T5 model.
        tokenizer: T5 tokenizer.
        data_loader: DataLoader for inference.
        
    Returns:
        Tuple of (predictions, references).
    """
    model.eval()
    predictions = []
    references = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
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


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main inference function."""
    
    print("=" * 60)
    print("T5 Slang Model Inference on California Plates")
    print("=" * 60)
    
    # Define paths
    model_path = MODEL_CHECKPOINTS_DIR / f"best_t5_ft_slang_model_{MODEL_TIMESTAMP}"
    input_csv = DATA_DIR / "cali.csv"
    
    timestamp = get_timestamp()
    output_csv = OUTPUTS_DIR / f"t5_ft_slang_predictions_on_cali_{timestamp}.csv"
    
    # Load input data
    print(f"\nLoading data from: {input_csv}")
    data = pd.read_csv(input_csv)
    data["plate"] = data["plate"].fillna("").astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)
    print(f"Loaded {len(data)} samples")

    # Load model and tokenizer
    print(f"\nLoading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(str(model_path))
    model = T5ForConditionalGeneration.from_pretrained(str(model_path)).to(DEVICE)
    
    # Create dataset and dataloader
    dataset = SlangDataset(data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Run inference
    predictions, references = run_inference(model, tokenizer, dataloader)

    # Get plate numbers
    plates = [data.iloc[idx]["plate"] for idx in range(len(dataset))]

    # Save results
    results_df = pd.DataFrame({
        "Plate": plates,
        "Predicted": predictions,
        "Reference": references,
    })
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Print examples
    print("\nExample predictions:")
    print("-" * 60)
    for i in range(min(5, len(predictions))):
        print(f"Plate: {plates[i]}")
        print(f"Predicted: {predictions[i][:100]}...")
        print(f"Reference: {references[i][:100]}...")
        print()


if __name__ == "__main__":
    main()
