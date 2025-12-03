import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import torch.nn as nn

## Define Parameters
MODEL_NAME_T5 = "t5-large"
MODEL_NAME_BERT = "bert-base-uncased"
BATCH_SIZE = 15
EPOCHS = 2
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Define Dataset
class VanityPlateDataset(Dataset):
    def __init__(self, dataframe, t5_tokenizer, bert_tokenizer, max_input_len, max_target_len):
        self.data = dataframe
        self.t5_tokenizer = t5_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        plate = self.data.iloc[idx]["plate"]
        comment = self.data.iloc[idx]["reviewer_comments"]

        # Tokenize inputs for T5 and BERT
        input_text = f"Translate vanity plate: {plate}"
        target_text = comment

        t5_inputs = self.t5_tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        bert_inputs = self.bert_tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.t5_tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": t5_inputs["input_ids"].squeeze(0),
            "attention_mask": t5_inputs["attention_mask"].squeeze(0),
            "bert_input_ids": bert_inputs["input_ids"].squeeze(0),
            "bert_attention_mask": bert_inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }

## Load Data
data = pd.read_csv('../data/cali.csv')
data = data.reset_index(drop=True)
data["plate"] = data["plate"].fillna("").astype(str)
data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)

## Prepare Tokenizers and Dataset
t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME_T5)
bert_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_BERT)

# Split dataset
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Create datasets
train_dataset = VanityPlateDataset(train_data, t5_tokenizer, bert_tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
val_dataset = VanityPlateDataset(val_data, t5_tokenizer, bert_tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
test_dataset = VanityPlateDataset(test_data, t5_tokenizer, bert_tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## Initialize Models
t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_T5).to(DEVICE)
bert_model = BertModel.from_pretrained(MODEL_NAME_BERT).to(DEVICE)

# Optimizer
optimizer = AdamW(list(t5_model.parameters()) + list(bert_model.parameters()), lr=LEARNING_RATE)

# Define linear projection for BERT embeddings
hidden_dim_t5 = t5_model.config.d_model  # T5 encoder hidden dimension
hidden_dim_bert = bert_model.config.hidden_size  # BERT hidden dimension
projection_layer = nn.Linear(hidden_dim_bert, hidden_dim_t5).to(DEVICE)

# Freeze BERT
bert_model.eval()

for epoch in range(EPOCHS):
    t5_model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader):
        # Extract batch data
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        bert_input_ids = batch["bert_input_ids"].to(DEVICE)
        bert_attention_mask = batch["bert_attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward pass through BERT (frozen)
        with torch.no_grad():
            bert_outputs = bert_model(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim_bert)

        # Project BERT embeddings to match T5 encoder hidden dimension
        print("Initial bert embedding shape", bert_embeddings.shape)

        bert_projected = projection_layer(bert_embeddings)  # Shape: (batch_size, seq_len, hidden_dim_t5)
        print("Projected bert embedding shape", bert_projected.shape)

        
        # Forward pass through T5 Encoder
        t5_encoder_outputs = t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        print("T5 embedding: ", t5_encoder_outputs.shape)

        # Fuse embeddings using element-wise sum
        fused_embeddings = t5_encoder_outputs + bert_projected  # Shape: (batch_size, seq_len, hidden_dim_t5)
        print("fused embedding: ", fused_embeddings.shape)

        # Pass fused embeddings to T5 decoder
        outputs = t5_model(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )

        # Compute loss
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss}")

# Save the model and tokenizer
t5_model.save_pretrained("../model_checkpoints/fused_t5_model_v0")
bert_model.save_pretrained("../model_checkpoints/fused_bert_model_v0")
t5_tokenizer.save_pretrained("../model_checkpoints/fused_t5_model_v0")
bert_tokenizer.save_pretrained("../model_checkpoints/fused_bert_model_v0")
