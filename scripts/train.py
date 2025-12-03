"""
Vanity Plate Interpretation Training Script
===========================================
Main training script using Flan-T5 (Google's instruction-tuned T5).

Model: google/flan-t5-base (default) - 250M parameters
       google/flan-t5-small - 80M parameters (faster)
       google/flan-t5-large - 780M parameters (better quality)

Features:
- Flan-T5 models (instruction-tuned, better than original T5)
- Train/Validation/Test splits with early stopping
- Integrated evaluation with multiple metrics
- LLM-as-a-Judge support (optional)
- Comprehensive logging

Usage:
    # Basic training
    python train.py
    
    # With smaller model
    python train.py --model google/flan-t5-small
    
    # Custom settings
    python train.py --model google/flan-t5-base --batch_size 8 --epochs 15 --lr 3e-5
    
    # With LLM-as-a-Judge evaluation
    python train.py --eval-with-llm --openai-key YOUR_KEY

Author: DSF Project Team
Last Updated: December 2025
"""

import argparse
import pandas as pd
import torch
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
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
EVAL_DIR = PROJECT_ROOT / "evaluations"

# Ensure directories exist
for d in [LOGS_DIR, MODEL_CHECKPOINTS_DIR, OUTPUTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "google/flan-t5-base"  # Flan-T5 base - best balance
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 128
SEED = 42
PATIENCE = 5

# Prompt template (Flan-T5 works better with natural language instructions)
PROMPT_TEMPLATE = "Interpret the meaning of this vanity license plate: {plate}"


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
    """
    Dataset for vanity plate interpretation.
    
    Formats input as natural language instruction for Flan-T5.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer,
        max_input_len: int = MAX_INPUT_LENGTH,
        max_target_len: int = MAX_TARGET_LENGTH,
        prompt_template: str = PROMPT_TEMPLATE
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
        meaning = str(self.data.iloc[idx]["reviewer_comments"])

        # Format with instruction template
        input_text = self.prompt_template.format(plate=plate)
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            meaning,
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

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False
) -> float:
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        if use_amp and scaler:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader: DataLoader, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
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


def generate_predictions(
    model,
    tokenizer,
    test_loader: DataLoader,
    device: torch.device,
    max_length: int = MAX_TARGET_LENGTH
) -> Tuple[List[str], List[str]]:
    """Generate predictions on test set."""
    model.eval()
    predictions, references = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)

    return predictions, references


# =============================================================================
# EVALUATION INTEGRATION
# =============================================================================

def run_evaluation(
    predictions_path: Path,
    use_llm_judge: bool = False,
    openai_key: Optional[str] = None,
    llm_samples: int = 50
) -> Dict:
    """Run evaluation on predictions file."""
    try:
        from evaluate import VanityPlateEvaluator
        
        df = pd.read_csv(predictions_path)
        
        evaluator = VanityPlateEvaluator(
            use_llm_judge=use_llm_judge,
            llm_backend="openai",
            llm_model="gpt-4o-mini",
            openai_api_key=openai_key
        )
        
        _, summary = evaluator.evaluate_dataframe(
            df,
            llm_judge_samples=llm_samples if use_llm_judge else None
        )
        
        return summary
    except ImportError:
        logging.warning("Evaluation module not available. Install required packages.")
        return {}
    except Exception as e:
        logging.warning(f"Evaluation failed: {e}")
        return {}


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Flan-T5 for vanity plate interpretation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default Flan-T5-base
    python train.py
    
    # Use smaller model for faster training
    python train.py --model google/flan-t5-small
    
    # Use larger model for better quality
    python train.py --model google/flan-t5-large --batch_size 2
    
    # Custom training settings
    python train.py --epochs 15 --batch_size 8 --lr 3e-5
    
    # With LLM-as-a-Judge evaluation
    python train.py --eval-with-llm --openai-key sk-...
    
Available Models:
    google/flan-t5-small   - 80M params, fastest
    google/flan-t5-base    - 250M params, balanced (default)
    google/flan-t5-large   - 780M params, best quality
        """
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    
    # Data arguments
    parser.add_argument("--data_file", type=str, default="cali.csv")
    parser.add_argument("--max_input_len", type=int, default=MAX_INPUT_LENGTH)
    parser.add_argument("--max_target_len", type=int, default=MAX_TARGET_LENGTH)
    
    # Evaluation arguments
    parser.add_argument("--eval-with-llm", action="store_true",
                        help="Run LLM-as-a-Judge evaluation after training")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key for LLM judge")
    parser.add_argument("--llm-samples", type=int, default=50,
                        help="Number of samples for LLM evaluation")
    
    # Other arguments
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision (faster on modern GPUs)")
    
    args = parser.parse_args()
    
    # Setup timestamp and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    log_file = LOGS_DIR / f"training_{model_short}_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Print banner
    logging.info("=" * 70)
    logging.info("🚗 VANITY PLATE INTERPRETATION TRAINING 🚗")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Max Epochs: {args.epochs}")
    logging.info(f"Early Stopping Patience: {args.patience}")
    logging.info("=" * 70)
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    logging.info("\n📊 Loading dataset...")
    data_path = DATA_DIR / args.data_file
    data = pd.read_csv(data_path)
    data["plate"] = data["plate"].fillna("").astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)
    logging.info(f"Loaded {len(data)} samples from {args.data_file}")
    
    # Load tokenizer and model
    logging.info(f"\n🤖 Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)
    
    logging.info(f"\n📂 Data splits:")
    logging.info(f"   Train: {len(train_data)}")
    logging.info(f"   Val:   {len(val_data)}")
    logging.info(f"   Test:  {len(test_data)}")
    
    # Create datasets
    train_dataset = VanityPlateDataset(train_data, tokenizer, args.max_input_len, args.max_target_len)
    val_dataset = VanityPlateDataset(val_data, tokenizer, args.max_input_len, args.max_target_len)
    test_dataset = VanityPlateDataset(test_data, tokenizer, args.max_input_len, args.max_target_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Training setup
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = MODEL_CHECKPOINTS_DIR / f"best_{model_short}_{timestamp}"
    
    # Training loop
    logging.info("\n🏋️ Starting training...")
    for epoch in range(args.epochs):
        logging.info(f"\n{'─'*50}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"{'─'*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE, 
            scaler=scaler, use_amp=args.use_amp
        )
        logging.info(f"📉 Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        logging.info(f"📊 Val Loss:   {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            logging.info(f"✅ New best model saved!")
        else:
            patience_counter += 1
            logging.info(f"⏳ No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                logging.info("🛑 Early stopping triggered!")
                break
    
    # Generate predictions
    logging.info("\n🔮 Generating test predictions...")
    predictions, references = generate_predictions(
        model, tokenizer, test_loader, DEVICE, args.max_target_len
    )
    
    # Save predictions
    test_plates = [test_data.iloc[i]["plate"] for i in range(len(test_dataset))]
    results_df = pd.DataFrame({
        "Plate": test_plates,
        "Predicted": predictions,
        "Reference": references
    })
    
    predictions_path = OUTPUTS_DIR / f"predictions_{model_short}_{timestamp}.csv"
    results_df.to_csv(predictions_path, index=False)
    logging.info(f"💾 Predictions saved to: {predictions_path}")
    
    # Run evaluation
    logging.info("\n📏 Running evaluation...")
    eval_summary = run_evaluation(
        predictions_path,
        use_llm_judge=args.eval_with_llm,
        openai_key=args.openai_key,
        llm_samples=args.llm_samples
    )
    
    # Print final summary
    logging.info("\n" + "=" * 70)
    logging.info("🎉 TRAINING COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"📁 Best Model: {best_model_path}")
    logging.info(f"📄 Predictions: {predictions_path}")
    logging.info(f"📝 Log File: {log_file}")
    
    if eval_summary:
        logging.info("\n📊 Evaluation Metrics:")
        logging.info("-" * 40)
        for key, value in eval_summary.items():
            if "_mean" in key:
                label = key.replace("_mean", "").upper()
                logging.info(f"   {label}: {value:.4f}")
    
    # Print example predictions
    logging.info("\n🔍 Example Predictions:")
    logging.info("-" * 40)
    for i in range(min(5, len(predictions))):
        logging.info(f"Plate: {test_plates[i]}")
        logging.info(f"Predicted: {predictions[i]}")
        logging.info(f"Reference: {references[i]}")
        logging.info("")
    
    # Save config
    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "best_val_loss": best_val_loss,
        "timestamp": timestamp,
        "evaluation": eval_summary
    }
    config_path = best_model_path / "training_config.json"
    best_model_path.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()

