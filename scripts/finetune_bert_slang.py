"""
BERT Slang Embedding Fine-tuning Script
=======================================
Fine-tunes a BERT model to generate meaningful embeddings for slang words
by learning to associate slang terms with their definitions.

Uses contrastive learning with cosine similarity loss to train the model
to produce similar embeddings for related (word, definition) pairs.

Dataset: Urban Dictionary word definitions (urbandict-word-defs.csv)
Model: BERT-base-uncased from Hugging Face Transformers

Features:
- Contrastive learning with InfoNCE-style loss
- Train/Validation split with reproducible seeding
- Early stopping to prevent overfitting
- Model checkpointing for best validation loss
- Embedding generation for inference

Usage:
    python finetune_bert_slang.py

Author: DSF Project Team
Last Updated: December 2025
"""

import torch
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm

# Import project configuration
try:
    from config import (
        DEVICE, DATA_DIR, LOGS_DIR, MODEL_CHECKPOINTS_DIR,
        LOG_FORMAT, LOG_DATE_FORMAT, BERTConfig
    )
except ImportError:
    # Fallback for standalone execution
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(__file__).parent.parent / "data"
    LOGS_DIR = Path(__file__).parent.parent / "logs"
    MODEL_CHECKPOINTS_DIR = Path(__file__).parent.parent / "model_checkpoints"
    LOG_FORMAT = "%(asctime)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
SEED = 42

# Data limiting (set to None for all data)
MAX_SAMPLES = 100000

# Early stopping configuration
PATIENCE = 2


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET CLASS
# =============================================================================

class UrbanDictionaryDataset(Dataset):
    """
    PyTorch Dataset for Urban Dictionary word-definition pairs.
    
    Tokenizes concatenated (word, definition) pairs for BERT embedding training.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing word and definition columns.
        tokenizer (BertTokenizer): Tokenizer for text encoding.
        max_length (int): Maximum sequence length.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_length: int
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: DataFrame with 'word' and 'definition' columns.
            tokenizer: Pre-trained BERT tokenizer.
            max_length: Maximum sequence length.
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Dictionary containing input_ids and attention_mask.
        """
        word = str(self.data.iloc[idx]["word"])
        definition = str(self.data.iloc[idx]["definition"])
        
        # Encode word-definition pair using BERT's [SEP] format
        encoding = self.tokenizer.encode_plus(
            word,
            definition,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class BERTForEmbedding(nn.Module):
    """
    BERT model wrapper for generating text embeddings.
    
    Uses the [CLS] token embedding as the sentence/word representation.
    
    Attributes:
        bert (BertModel): Pre-trained BERT model.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the pre-trained BERT model to load.
        """
        super(BERTForEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from the tokenizer.
            attention_mask: Attention mask for padding.
            
        Returns:
            [CLS] token embeddings of shape (batch_size, hidden_dim).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Return the [CLS] token embedding (first token)
        return outputs.last_hidden_state[:, 0, :]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_contrastive_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute InfoNCE-style contrastive loss for batch of embeddings.
    
    This loss encourages similar items to have similar embeddings
    and dissimilar items to have different embeddings.
    
    Args:
        embeddings: Batch of embeddings of shape (batch_size, hidden_dim).
        
    Returns:
        Scalar loss value.
    """
    # Compute pairwise cosine similarity
    similarity = torch.cosine_similarity(
        embeddings[:, None, :], 
        embeddings[None, :, :], 
        dim=-1
    )
    
    # InfoNCE-style loss (each sample is its own positive, others are negatives)
    loss = -torch.log(
        torch.exp(similarity) / torch.exp(similarity).sum(dim=-1, keepdim=True)
    ).mean()
    
    return loss


def train_epoch(
    model: BERTForEmbedding,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: BERT embedding model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for parameter updates.
        device: Device to run training on.
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(input_ids, attention_mask)
        
        # Compute contrastive loss
        loss = compute_contrastive_loss(embeddings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: BERTForEmbedding,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Validate the model on the validation set.
    
    Args:
        model: BERT embedding model to validate.
        val_loader: DataLoader for validation data.
        device: Device to run validation on.
        
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            embeddings = model(input_ids, attention_mask)
            loss = compute_contrastive_loss(embeddings)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(
    model: BERTForEmbedding,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    num_epochs: int,
    patience: int,
    model_save_path: Path
) -> Path:
    """
    Full training loop with early stopping.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        device: Device to train on.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        model_save_path: Path to save the best model.
        
    Returns:
        Path to the saved best model.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        avg_train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation
        avg_val_loss = validate(model, val_loader, device)
        
        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"✓ New best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model_save_path


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def generate_embedding(
    model: BERTForEmbedding,
    tokenizer: BertTokenizer,
    word: str,
    device: torch.device
) -> np.ndarray:
    """
    Generate embedding for a single slang word.
    
    Args:
        model: Trained BERT embedding model.
        tokenizer: BERT tokenizer.
        word: Slang word to embed.
        device: Device to run inference on.
        
    Returns:
        Embedding vector as numpy array.
    """
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            word,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        embedding = model(input_ids, attention_mask)
        
    return embedding.cpu().numpy().squeeze()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Generate timestamp for this run
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    log_filename = LOGS_DIR / f"training_logs_bert_slang_{time_now}.txt"
    logging.basicConfig(
        filename=str(log_filename),
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    logging.info("=" * 60)
    logging.info("BERT Slang Embedding Fine-tuning")
    logging.info("=" * 60)
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Max Epochs: {EPOCHS}")
    
    # Load and preprocess dataset
    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_DIR / "urbandict-word-defs.csv", on_bad_lines='skip')
    df = df.dropna()
    df = df[["word", "definition"]]
    
    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
    
    logging.info(f"Loaded {len(df)} samples")
    
    # Initialize tokenizer and model
    logging.info("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BERTForEmbedding(MODEL_NAME).to(DEVICE)
    
    # Create dataset and split
    full_dataset = UrbanDictionaryDataset(df, tokenizer, max_length=MAX_LENGTH)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    logging.info(f"Train: {train_size}, Validation: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Model save path
    model_save_path = MODEL_CHECKPOINTS_DIR / "best_bert_slang_finetuned_model.pth"
    
    # Train the model
    logging.info("Starting training...")
    best_model_path = train(
        model, train_loader, val_loader, optimizer, 
        DEVICE, EPOCHS, PATIENCE, model_save_path
    )
    
    # Load best model for example inference
    logging.info("Loading best model for inference examples...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Generate example embeddings
    example_words = ["yolo", "lit", "fomo", "salty", "ghosting", "vibing"]
    logging.info("\nExample embeddings:")
    
    for word in example_words:
        embedding = generate_embedding(model, tokenizer, word, DEVICE)
        logging.info(f"'{word}': shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Training completed successfully!")
    logging.info(f"Model saved to: {best_model_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
