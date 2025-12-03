"""
Modern Model Training Script for Vanity Plate Interpretation
=============================================================
Supports multiple modern seq-to-seq models as alternatives to T5-large.

Supported Models:
- Flan-T5 (small, base, large) - Recommended!
- T5-efficient (tiny, mini, small, base)
- BART (base, large)
- mT5 (small, base) - Multilingual

Usage:
    python train_modern.py --model flan-t5-base
    python train_modern.py --model flan-t5-small --epochs 10
    python train_modern.py --model bart-base

Author: AMS 560
Last Updated: December 2025
"""

import argparse
import pandas as pd
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "model_checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for d in [LOGS_DIR, MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {
    # Flan-T5 Models (Recommended - instruction-tuned)
    "flan-t5-small": {
        "hf_name": "google/flan-t5-small",
        "params": "80M",
        "description": "Instruction-tuned T5, great for small GPUs"
    },
    "flan-t5-base": {
        "hf_name": "google/flan-t5-base",
        "params": "250M",
        "description": "Best balance of performance and efficiency"
    },
    "flan-t5-large": {
        "hf_name": "google/flan-t5-large",
        "params": "780M",
        "description": "High performance, needs more VRAM"
    },
    
    # T5-efficient Models (Optimized for speed)
    "t5-efficient-tiny": {
        "hf_name": "google/t5-efficient-tiny",
        "params": "17M",
        "description": "Ultra-small, fastest inference"
    },
    "t5-efficient-mini": {
        "hf_name": "google/t5-efficient-mini",
        "params": "31M",
        "description": "Very small and fast"
    },
    "t5-efficient-small": {
        "hf_name": "google/t5-efficient-small",
        "params": "60M",
        "description": "Small and efficient"
    },
    "t5-efficient-base": {
        "hf_name": "google/t5-efficient-base",
        "params": "220M",
        "description": "Efficient base model"
    },
    
    # Original T5 Models
    "t5-small": {
        "hf_name": "t5-small",
        "params": "60M",
        "description": "Original T5 small"
    },
    "t5-base": {
        "hf_name": "t5-base",
        "params": "220M",
        "description": "Original T5 base"
    },
    "t5-large": {
        "hf_name": "t5-large",
        "params": "770M",
        "description": "Original T5 large (current default)"
    },
    
    # BART Models
    "bart-base": {
        "hf_name": "facebook/bart-base",
        "params": "140M",
        "description": "BART encoder-decoder, good for generation"
    },
    "bart-large": {
        "hf_name": "facebook/bart-large",
        "params": "400M",
        "description": "Larger BART model"
    },
    
    # Multilingual
    "mt5-small": {
        "hf_name": "google/mt5-small",
        "params": "300M",
        "description": "Multilingual T5"
    },
}


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# =============================================================================
# DATASET CLASS
# =============================================================================

class VanityPlateDataset(Dataset):
    """Dataset for vanity plate interpretation."""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer,
        max_input_len: int = 32,
        max_target_len: int = 128,
        prompt_template: str = "Interpret this vanity plate: {plate}"
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        plate = str(self.data.iloc[idx]["plate"])
        comment = str(self.data.iloc[idx]["reviewer_comments"])

        input_text = self.prompt_template.format(plate=plate)
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            comment,
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
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
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
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def validate(model, val_loader, device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
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
            total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


def generate_predictions(model, tokenizer, test_loader, device, max_length=128):
    """Generate predictions on test set."""
    model.eval()
    predictions, references = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
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
    parser = argparse.ArgumentParser(
        description="Train modern seq-to-seq models for vanity plate interpretation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  Flan-T5 (Recommended):
    --model flan-t5-small   (80M params, fast)
    --model flan-t5-base    (250M params, balanced)
    --model flan-t5-large   (780M params, best quality)
    
  T5-Efficient (Speed optimized):
    --model t5-efficient-tiny  (17M params, fastest)
    --model t5-efficient-small (60M params)
    
  BART:
    --model bart-base  (140M params)
    --model bart-large (400M params)

Examples:
    python train_modern.py --model flan-t5-small --epochs 10 --batch_size 8
    python train_modern.py --model flan-t5-base --lr 3e-5
        """
    )
    
    parser.add_argument(
        "--model", type=str, default="flan-t5-base",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train (default: flan-t5-base)"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_input_len", type=int, default=32, help="Max input length")
    parser.add_argument("--max_target_len", type=int, default=128, help="Max target length")
    parser.add_argument("--data_file", type=str, default="cali.csv", help="Dataset file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Get model info
    model_info = MODEL_REGISTRY[args.model]
    model_name = model_info["hf_name"]
    
    # Setup timestamp and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.replace("-", "_")
    log_file = LOGS_DIR / f"training_{model_short}_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Print configuration
    logging.info("=" * 70)
    logging.info("VANITY PLATE INTERPRETATION - MODERN MODEL TRAINING")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model} ({model_info['params']} parameters)")
    logging.info(f"HuggingFace: {model_name}")
    logging.info(f"Description: {model_info['description']}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Max Epochs: {args.epochs}")
    logging.info("=" * 70)
    
    # Load data
    logging.info("\nLoading dataset...")
    data_path = DATA_DIR / args.data_file
    data = pd.read_csv(data_path)
    data["plate"] = data["plate"].fillna("").astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)
    logging.info(f"Loaded {len(data)} samples from {args.data_file}")
    
    # Load tokenizer and model
    logging.info(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)
    
    # Create datasets
    train_dataset = VanityPlateDataset(train_data, tokenizer, args.max_input_len, args.max_target_len)
    val_dataset = VanityPlateDataset(val_data, tokenizer, args.max_input_len, args.max_target_len)
    test_dataset = VanityPlateDataset(test_data, tokenizer, args.max_input_len, args.max_target_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"\nData splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = MODEL_CHECKPOINTS_DIR / f"best_{model_short}_{timestamp}"
    
    logging.info("\nStarting training...")
    for epoch in range(args.epochs):
        logging.info(f"\n{'='*50}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        logging.info(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            logging.info(f"✓ New best model saved!")
        else:
            patience_counter += 1
            logging.info(f"No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered!")
                break
    
    # Generate test predictions
    logging.info("\nGenerating test predictions...")
    predictions, references = generate_predictions(
        model, tokenizer, test_loader, DEVICE, args.max_target_len
    )
    
    # Save results
    test_plates = [test_data.iloc[i]["plate"] for i in range(len(test_dataset))]
    results_df = pd.DataFrame({
        "Plate": test_plates,
        "Predicted": predictions,
        "Reference": references
    })
    
    output_path = OUTPUTS_DIR / f"predictions_{model_short}_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    logging.info(f"\n{'='*70}")
    logging.info("TRAINING COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"Best model: {best_model_path}")
    logging.info(f"Predictions: {output_path}")
    logging.info(f"Log file: {log_file}")
    
    # Show examples
    logging.info("\nExample Predictions:")
    logging.info("-" * 50)
    for i in range(min(5, len(predictions))):
        logging.info(f"Plate: {test_plates[i]}")
        logging.info(f"Predicted: {predictions[i]}")
        logging.info(f"Reference: {references[i]}")
        logging.info("")


if __name__ == "__main__":
    main()

