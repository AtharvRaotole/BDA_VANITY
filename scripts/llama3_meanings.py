import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
from tqdm import tqdm  # For progress monitoring

# Specify the model ID
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Load the config
config = AutoConfig.from_pretrained(model_id, token=True)
config.tie_word_embeddings = False  # Disable weight tying in the config

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=True,
)
print("Model and tokenizer loaded successfully.")

def generate_meaning_with_model(plates, inputs_text):
    """
    Generate meanings for vanity plates using LLaMA 3.1 in batches.
    """
    print(f"Generating meanings for a batch of {len(plates)} plates...")

    # Tokenize the batch of input texts
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Check if inputs exceed the model's max token length
    if inputs.input_ids.shape[1] > tokenizer.model_max_length:
        print("Warning: Input truncated. Consider shortening prompts.")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=55,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # Suppress warning
    )

    # Decode the generated tokens and strip the prompts
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

    results = []
    for text, prompt in zip(generated_texts, inputs_text):
        # Find the first occurrence of the meaning after the prompt
        meaning_start = text.lower().replace(prompt.lower(), '').strip()
        results.append(meaning_start)
    print(f"Batch generation completed. Generated {len(results)} meanings.")
    return results

def process_csv_with_model(input_file, output_file, batch_size=16):
    """
    Process an input CSV file containing vanity plate information,
    generate meanings for each plate in batches, and save the output to a new file.
    """
    print(f"Reading input CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"CSV file loaded successfully. Total records: {len(df)}")

    # Ensure the necessary columns exist
    required_columns = {'plate', 'status', 'customer_meaning', 'reviewer_comments'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    print("Required columns found in the CSV.")

    # Prepare inputs for generation
    plates = []
    input_texts = []
    meanings = []
    batch_count = 0  # Counter to track processed batches

    total_batches = (len(df) + batch_size - 1) // batch_size

    # Use tqdm for progress tracking
    print("Starting batch processing...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        plate = row['plate']
        if row['status'] == 'Y':
            input_text = row['customer_meaning']
        else:
            input_text = row['reviewer_comments']

        # Build prompt
        prompt = (
            f"You are an assistant to a Department of Motor Vehicles (DMV) Personalized License Plate reviewer. "
            "Your task is to analyze and explain the meaning of the license plate in 1-2 lines going into details if its meaning:\n"
            "   - Contains a sexual connotation or term of lust or depravity.\n"
            "   - Is vulgar, contemptuous, prejudiced or hostile.\n"
            "   - Includes profanity, obscenity or repulsive language.\n"
            "   - Has a negative connotation for any specific group.\n"
            "   - Misrepresents a law enforcement entity.\n"
            "   - Uses foreign or slang terms inappropriately.\n"
            "   - Is a phonetic or mirrored spelling of any of the above.\n"
            "   - Celebrates achievements, milestones, or personal success.\n"
            "   - Encourages positivity, motivation, or inspiration.\n"
            "   - Reflects hobbies, interests, or fandoms.\n"
            "Provide specific details, nuances and potential hidden meanings of the plate relying on the Possible Meaning given below.\n\n"
            f"Plate: {plate} Possible Meaning: {input_text}\n"
            "You are encouraged to mention expletives or details of how and why a plate has a particular connotation/meaning. Only give the meaning without repeating this prompt."
        )

        plates.append(plate)
        input_texts.append(prompt)

        # Process in batches
        if len(input_texts) == batch_size or idx == len(df) - 1:
            print(f"\nProcessing batch {batch_count + 1}/{total_batches}...")
            batch_meanings = generate_meaning_with_model(plates, input_texts)
            meanings.extend(batch_meanings)
            plates = []
            input_texts = []
            batch_count += 1

    print("Adding generated meanings to the DataFrame.")
    df = df.head(len(meanings))  # Only include rows that were processed
    df['meaning'] = pd.Series(meanings).reindex_like(df)

    print(f"Saving updated DataFrame to output file: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"File saved successfully to {output_file}")

# Example usage
input_csv = "/home/vivora/data/cali.csv"  # Input CSV file path
output_csv = "/home/vivora/data/vanity_plates_llama_meanings.csv"  # Output CSV file path
process_csv_with_model(input_csv, output_csv, batch_size=16)






# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# import pandas as pd
# from tqdm import tqdm  # For progress monitoring

# # Specify the model ID
# model_id = "meta-llama/Llama-3.1-8B-Instruct"

# # Load the config first
# config = AutoConfig.from_pretrained(model_id, token=True)
# config.tie_word_embeddings = False  # Disable weight tying in the config

# # Load the tokenizer and model
# print("Loading tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     config=config,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     token=True,
# )
# print("Model and tokenizer loaded successfully.")

# def generate_meaning_with_model(plates, inputs_text):
#     """
#     Generate meanings for vanity plates using LLaMA 3.1 in batches.
#     """
#     print(f"Generating meanings for a batch of {len(plates)} plates...")

#     # Tokenize the batch of input texts
#     inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
#     # Generate output
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=40,
#         temperature=0.2,
#         top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id  # Suppress warning
#     )

#     # Decode the generated tokens and strip the prompts
#     results = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
#     print(f"Batch generation completed. Generated {len(results)} meanings.")
#     return results

# def process_csv_with_model(input_file, output_file, batch_size=16):
#     """
#     Process an input CSV file containing vanity plate information,
#     generate meanings for each plate in batches, and save the output to a new file.
#     """
#     print(f"Reading input CSV file: {input_file}")
#     df = pd.read_csv(input_file)
#     print(f"CSV file loaded successfully. Total records: {len(df)}")

#     # Ensure the necessary columns exist
#     required_columns = {'plate', 'status', 'customer_meaning', 'reviewer_comments'}
#     if not required_columns.issubset(df.columns):
#         raise ValueError(f"CSV must contain columns: {required_columns}")
#     print("Required columns found in the CSV.")

#     # Prepare inputs for generation
#     plates = []
#     input_texts = []
#     meanings = []

#     # Use tqdm for progress tracking
#     print("Starting batch processing...")
#     total_batches = (len(df) + batch_size - 1) // batch_size
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
#         plate = row['plate']
#         if row['status'] == 'Y':
#             input_text = row['customer_meaning']
#         else:
#             input_text = row['reviewer_comments']

#         plates.append(plate)
#         input_texts.append(
#             f"Given the vanity plate '{plate}', explain its meaning in 1-2 lines. Input: {input_text}\n"
#         )

#         # Process in batches
#         if len(input_texts) == batch_size or idx == len(df) - 1:
#             print(f"\nProcessing batch {len(meanings) // batch_size + 1}/{total_batches}...")
#             batch_meanings = generate_meaning_with_model(plates, input_texts)
#             meanings.extend(batch_meanings)
#             plates = []
#             input_texts = []

#     print("All batches processed. Adding generated meanings to the DataFrame.")
#     df['meaning'] = meanings

#     print(f"Saving updated DataFrame to output file: {output_file}")
#     df.to_csv(output_file, index=False)
#     print(f"File saved successfully to {output_file}")

# # Example usage
# input_csv = "/home/vivora/data/cali.csv"  # Input CSV file path
# output_csv = "vanity_plates_with_meaning.csv"  # Output CSV file path
# process_csv_with_model(input_csv, output_csv, batch_size=16)
