import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sentence_transformers import SentenceTransformer
from torch.nn import CosineEmbeddingLoss
import logging
from transformers import DataCollatorForSeq2Seq

# Initialize log filename with timestamp
time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"../logs/training_logs_new_bert_slang_finetuned_{time_now}.txt"

# Set up logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load dataset
df = pd.read_csv("../data/urbandict-word-defs.csv", on_bad_lines='skip')
df = df.dropna().drop_duplicates()
df = df.head(100)
logging.info(f"Dataset loaded with {len(df)} entries.")

# Split into train and validation
train_texts, val_texts, train_defs, val_defs = train_test_split(
    df['word'].tolist(), df['definition'].tolist(), test_size=0.2, random_state=42
)
logging.info(f"Train-Validation split completed: {len(train_texts)} train and {len(val_texts)} validation.")

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({'word': train_texts, 'definition': train_defs})
val_dataset = Dataset.from_dict({'word': val_texts, 'definition': val_defs})

# Initialize tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the slang word
def tokenize_function(examples):
    return tokenizer(examples['word'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])




# Load pre-trained Sentence Embedding Model for definitions
definition_model = SentenceTransformer('all-MiniLM-L6-v2')
definition_model.eval()  # Freeze the model

# Contrastive Loss
loss_fn = CosineEmbeddingLoss()

# Convert 'definition_emb' to a list of Python-native types (from PyTorch tensors)
train_defs_emb = definition_model.encode(train_defs)
val_defs_emb = definition_model.encode(val_defs)

# Add the definition embeddings as columns
train_dataset = train_dataset.add_column("definition_emb", train_defs_emb.tolist())
val_dataset = val_dataset.add_column("definition_emb", val_defs_emb.tolist())

# Update dataset format to include 'definition_emb'
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "definition_emb"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "definition_emb"])

logging.info(f"Train dataset columns: {train_dataset.column_names}")
logging.info(f"Validation dataset columns: {val_dataset.column_names}")

# Training setup
training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,  # Load the best model based on validation loss
    metric_for_best_model="loss",  # Best model determined by validation loss
    logging_first_step=True,  # Log after the first step
    remove_unused_columns=False,  
)

from transformers import Trainer
import torch
import numpy as np

# Define the model
class SlangEmbeddingModel(nn.Module):
    def __init__(self, model_name, embedding_dim=768):
        super(SlangEmbeddingModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  # Pre-trained encoder
        self.linear = nn.Linear(embedding_dim, embedding_dim)  # Optional transformation layer
        self.definition_linear = nn.Linear(384, embedding_dim)  # Project definition embeddings to 768
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output  # CLS token embedding
        return self.linear(embeddings)  # Return transformed embeddings

# Custom Trainer with Contrastive Loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = inputs["input_ids"].device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        definition_emb = inputs.get("definition_emb").to(device)

        # Get slang word embeddings from the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Project definition embeddings to match model's output size (768)
        definition_emb = model.definition_linear(definition_emb)

        # Calculate cosine similarity loss
        target = torch.ones(outputs.size(0)).to(device)  # Positive pairs
        loss = loss_fn(outputs, definition_emb, target)
        
        if return_outputs:
            return loss, outputs
        return loss

    # Override the prediction step to avoid passing definition_emb to the model
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Remove definition_emb from inputs passed to the model
        inputs = {key: value for key, value in inputs.items() if key != "definition_emb"}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    # Define metrics for evaluation
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        loss = self.compute_loss(self.model, {"input_ids": logits, "attention_mask": labels})
        return {"eval_loss": loss.item()}

# Override the default data collator to include `definition_emb` as part of the batch
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        # First, tokenize the input examples
        batch = {
            'input_ids': torch.stack([feature['input_ids'] for feature in features]),
            'attention_mask': torch.stack([feature['attention_mask'] for feature in features]),
            'definition_emb': torch.stack([feature['definition_emb'] for feature in features])
        }
        return batch

trainer = CustomTrainer(
    model=SlangEmbeddingModel(model_name),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping
    data_collator=CustomDataCollator(tokenizer),  # Use the custom data collator
)

# Start training and log progress
logging.info("Training started...")
trainer.train()
logging.info("Training completed.")

# Save the model checkpoint and tokenizer
trainer.save_model("../results/bert_ft_model")
tokenizer.save_pretrained("../results/bert_ft_model")
logging.info("Model and tokenizer saved.")

# Load the best model for evaluation
best_model = SlangEmbeddingModel(model_name)
best_model.load_state_dict(torch.load('../results/bert_ft_model/pytorch_model.bin'))
best_model.eval()
logging.info("Best model loaded for evaluation.")

# Example usage: Get embedding for a slang word after fine-tuning
def get_slang_embedding(slang_word, model, tokenizer):
    inputs = tokenizer(slang_word, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        embedding = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    return embedding.squeeze(0).numpy()

# Example usage: Get embedding for a slang word
slang_word = "YOLO"
embedding = get_slang_embedding(slang_word, best_model, tokenizer)
logging.info(f"Embedding for slang word '{slang_word}': {embedding}")
print(embedding)
