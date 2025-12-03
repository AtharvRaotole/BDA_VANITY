import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

time_now = "20241130_141308"
# Define parameters
MODEL_PATH = f"../model_checkpoints/best_t5_ft_slang_abb_model_{time_now}"  # Replace with your saved model path
INPUT_CSV = "../data/cali.csv"  # Replace with your input CSV file
OUTPUT_CSV = f"../outputs/t5_ft_slang_abb_predictions_on_cali_{time_now}.csv"
BATCH_SIZE = 4
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Dataset class
class SlangDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len, max_target_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data.iloc[idx]["plate"]
        definition = self.data.iloc[idx]["reviewer_comments"]

        # Tokenize inputs
        input_text = f"Translate this word: {word}"
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

# Load input data
print("Loading input data...")
data = pd.read_csv(INPUT_CSV)
data["plate"] = data["plate"].fillna("").astype(str)
data["reviewer_comments"] = data["reviewer_comments"].fillna("").astype(str)

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

# Set the random seed for reproducibility
SEED = 42

dataset = SlangDataset(data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Perform inference
print("Performing inference...")
model.eval()
test_predictions = []
test_references = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=MAX_TARGET_LENGTH)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        test_predictions.extend(preds)
        test_references.extend(refs)

# Retrieve plate numbers from the test dataset
test_plate_numbers = [data.iloc[idx]["plate"] for idx in range(len(dataset))]


# Prepare results for saving
results = {
    "Plate": test_plate_numbers,
    "Predicted": test_predictions,
    "Reference": test_references,
}

# Create a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"Results saved to {OUTPUT_CSV}")
