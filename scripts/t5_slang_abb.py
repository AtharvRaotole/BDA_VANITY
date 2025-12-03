import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.optim import AdamW
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split


## Define params
MODEL_NAME = "t5-large"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Collate Dataset
class VanityPlateDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len, max_target_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        plate = self.data.iloc[idx]["plate"]
        comment = self.data.iloc[idx]["reviewer_comments"]

        # Tokenize inputs
        input_text = f"Translate vanity plate: {plate}"
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

## Load CSV
data = pd.read_csv('../data/cali_slang_abb.csv')
data = data.reset_index(drop=True)  
data["plate"] = data["plate"].fillna("").astype(str)
data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)

## Prepare Dataset
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
dataset = VanityPlateDataset(data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

# Set the random seed for reproducibility
SEED = 42

# Split the data into train, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# Create Dataset instances for each split
train_dataset = VanityPlateDataset(train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
val_dataset = VanityPlateDataset(val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
test_dataset = VanityPlateDataset(test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, resume_download=True).to(DEVICE)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Generate a timestamped log filename
time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
time_now += "_slang_abb"
log_filename = f"../logs/training_logs_{time_now}.txt"

# Set up logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("\n\n")
# # Training Loop
# model.train()
# for epoch in range(EPOCHS):
#     epoch_loss = 0
#     logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
#     for batch in tqdm(train_loader):
#         input_ids = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         labels = batch["labels"].to(DEVICE)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         epoch_loss += loss.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     avg_train_loss = epoch_loss / len(train_loader)
#     logging.info(f"Training Loss: {avg_train_loss}")

# logging.info("\n\n")

# # Validation Loop
# model.eval()
# val_loss = 0
# with torch.no_grad():
#     for batch in tqdm(val_loader):
#         input_ids = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         labels = batch["labels"].to(DEVICE)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         val_loss += outputs.loss.item()

# avg_val_loss = val_loss / len(val_loader)
# logging.info(f"Validation Loss: {avg_val_loss}")

# logging.info("\n\n")

# # Save Model
# model.save_pretrained(f"../model_checkpoints/vanity_plate_model_{time_now}")
# tokenizer.save_pretrained(f"../model_checkpoints/vanity_plate_model_{time_now}")

# logging.info(f"Model and tokenizer saved to 'vanity_plate_model_{time_now}'.")

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Early Stopping Parameters
patience = 5  # Number of epochs with no improvement after which training stops
min_delta = 1e-4  # Minimum change in the validation loss to qualify as an improvement
best_val_loss = float("inf")  # Initialize to infinity for comparison
patience_counter = 0  # Counter for epochs with no improvement
best_model_path = f"../model_checkpoints/best_vanity_plate_model_{time_now}"  # Path to save the best model

for epoch in range(EPOCHS):
    epoch_loss = 0
    logging.info(f"Epoch {epoch + 1}/{EPOCHS}")

    # Training Loop
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_loader)
    logging.info(f"Training Loss: {avg_train_loss}")

    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Validation Loss: {avg_val_loss}")

    # Early Stopping Check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        logging.info(f"Best model saved with validation loss: {avg_val_loss}")
    else:
        patience_counter += 1
        logging.info(f"No improvement for {patience_counter} epochs.")

    if patience_counter >= patience:
        logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# # After the loop ends, save the final model and tokenizer
# final_model_path = f"../model_checkpoints/vanity_plate_model_final_{time_now}"
# model.save_pretrained(final_model_path)
# tokenizer.save_pretrained(final_model_path)

logging.info(f"Best model and tokenizer saved to '{best_model_path}'.")

logging.info("\n\n")

# Testing Loop
model.eval()
test_predictions = []
test_references = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=MAX_TARGET_LENGTH)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        test_predictions.extend(preds)
        test_references.extend(refs)

# Retrieve plate numbers from the test dataset
test_plate_numbers = [test_data.iloc[idx]["plate"] for idx in range(len(test_dataset))]

# Prepare results for saving
results = {
    "Plate": test_plate_numbers,
    "Predicted": test_predictions,
    "Reference": test_references,
}

# Create a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(f"../outputs/vanity_plate_predictions.csv_{time_now}", index=False)
logging.info(f"Results saved to 'vanity_plate_predictions.csv_{time_now}'.")
