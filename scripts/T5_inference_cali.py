import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# Set the random seed globally for numpy and pandas
np.random.seed(42)

# Read the datasets
# cali_df = pd.read_csv('/home/vivora/data/cali.csv')
cali_df = pd.read_csv('/home/vivora/data/new_york.csv')

cali_df = cali_df[['plate','status']]

# Shuffle the entire dataset with a fixed random seed
cali_df = cali_df.sample(frac=1, random_state=42).reset_index(drop=True)

cali_df['plate'] = cali_df['plate'].astype(str)
cali_df['status'] = cali_df['status'].astype(str)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model and data paths (adjust these for your environment)
# MODEL_DIRECTORY = '/home/vivora/finetuned_T5_v2'
MODEL_DIRECTORY = '/home/vivora/finetuned_T5_llama_rc'
OUTPUT_BASE_PATH = '/home/vivora/data'

# Model hyperparameters
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

def load_model_and_tokenizer(model_dir):
    """
    Load T5 model and tokenizer
    """
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_plate_meaning(plate, model, tokenizer):
    """
    Predict meaning for a single plate
    """
    try:
        # Prepare the input text matching training format
        input_text = f"Translate vanity plate: {plate}"
        
        # Tokenize the input
        inputs = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                max_length=MAX_TARGET_LENGTH
            )
        
        # Decode the prediction
        meaning = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return meaning
    except Exception as e:
        print(f"Error processing plate {plate}: {e}")
        return "ERROR"

def process_vanity_plates(df, model, tokenizer, test_mode=True):
    """
    Process vanity plates with option for testing or full dataset
    """
    # Create a copy of the DataFrame
    processed_df = df.copy()
    
    # Determine number of rows to process
    if test_mode:
        processed_df = processed_df.head(10)
        output_filename = 'test_vanity_plates_meanings_T5_llama_rc.csv'
    else:
        # output_filename = 'cali_vanity_plates_meanings_T5_llama_rc.csv'
        output_filename = 'new_york_vanity_plates_meanings_T5_llama_rc.csv'
    
    # Use tqdm for progress tracking
    tqdm.pandas()
    
    # Apply prediction
    processed_df['predicted_meaning'] = processed_df['plate'].progress_apply(
        lambda x: predict_plate_meaning(x, model, tokenizer)
    )
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_BASE_PATH, output_filename)
    processed_df.to_csv(output_path, index=False)
    
    print(f"\nSaved results to: {output_path}")
    print("\nFirst few rows with predictions:")
    print(processed_df[['plate', 'predicted_meaning']].head())
    
    return processed_df

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_DIRECTORY)
    
if model is None or tokenizer is None:
    print("Failed to load model. Exiting.")
    
# Set model to evaluation mode
model.eval()
    
# Test mode: process first 10 rows
# test_results = process_vanity_plates(vanity_plates_df, model, tokenizer, test_mode=True)
    
# Uncomment the following line when ready to process full dataset on GPU server
full_results = process_vanity_plates(cali_df, model, tokenizer, test_mode=False)
print("Full Results saved!")
