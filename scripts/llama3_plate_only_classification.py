import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
import os
import datetime

# Specify the model ID
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Load the config
config = AutoConfig.from_pretrained(model_id, token=True)
config.tie_word_embeddings = False

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
    token=True
)

print("Model and tokenizer loaded successfully.")

def extract_assistant_response(full_output, input_prompt):
    # Remove the input prompt from the beginning
    response = full_output.replace(input_prompt, '', 1).strip()
    
    # Split the response by role indicators
    parts = response.split('assistant')
    
    if len(parts) > 1:
        # Take the last part (assistant's response)
        assistant_response = parts[-1].strip()
        
        # Remove any remaining role indicators
        assistant_response = assistant_response.replace('user', '').replace('assistant', '').strip()
        
        return assistant_response
    else:
        return response

#Removed below line from prompt to relax the decisions made by llama - you may add it back:
# #Remember, you have discretion to REJECT any plate that may be considered inappropriate or offensive, even if it doesn't explicitly violate a rule. Err on the side of caution to avoid potential public complaints.
def generate_decision(plate):
    
#     system_prompt = f"""You are an AI assistant for a Department of Motor Vehicles (DMV) tasked with reviewing personalized license plate requests. Your job is to evaluate plate requests and determine if they should be ACCEPTED or REJECTED based on DMV guidelines.\n
# Follow these steps:
# Check if the requested plate meets character count limits (typically 2-8 characters).\n
# Verify only allowed characters are used (A-Z, 0-9, spaces, limited special characters).\n
# Analyze the plate text for any prohibited content:\n
# •Obscenity, vulgarity, or offensive language\n
# •Sexual or excretory references\n
# •Drug references\n
# •Violent content\n
# •Discriminatory or derogatory terms\n
# •Misleading official terms\n
# •Copyrighted or trademarked phrases\n
# •Confusing letter/number combinations\n
# Consider if the plate could be interpreted as inappropriate when viewed on a vehicle.\n
# Determine if the plate should be ACCEPTED or REJECTED and ONLY give a single decision.\n
# Remember, you have discretion to REJECT any plate that may be considered inappropriate or offensive, even if it doesn't explicitly violate a rule. Err on the side of caution to avoid potential public complaints.\n

# You must respond with ONLY "ACCEPTED" or "REJECTED". Do not provide any explanation or additional text in your response.
# Output Format:
# [Your decision]\n
# """


#     system_prompt = f"""You are an AI assistant for a Department of Motor Vehicles (DMV) tasked with reviewing personalized license plate requests. Your job is to evaluate plate requests and determine if they should be ACCEPTED or REJECTED based on DMV guidelines.

# Follow these steps:
# 1. Check if the requested plate meets character count limits (typically 2-8 characters).
# 2. Verify only allowed characters are used (A-Z, 0-9, spaces, limited special characters).
# 3. Analyze the plate text for any prohibited content:
#    - Obscenity, vulgarity, or offensive language
#    - Sexual or excretory references
#    - Drug references
#    - Violent content
#    - Discriminatory or derogatory terms
#    - Misleading official terms
#    - Copyrighted or trademarked phrases
#    - Confusing letter/number combinations
# 4. Alternatively, analyze if it has a positive or neutral connotation such as:
#     - Demonstrates creativity, humor, or cleverness.\n"
#     - Celebrates achievements, milestones, or personal success.\n"
#     - Conveys love, affection, or other positive emotions.\n"
#     - Promotes cultural pride, heritage, or traditions.\n"
#     - Encourages positivity, motivation, or inspiration.\n"
#     - Reflects hobbies, interests, or fandoms.\n"
#     - Advocates for a social cause or inclusivity.\n"
#     - Highlights loyalty or pride in a community or team.\n"
#     - Represents playful self-expression or identity.\n\n"
# 5. Consider if the plate could be interpreted as inappropriate when viewed on a vehicle.
# 6. Determine if the plate should be ACCEPTED or REJECTED.

# DMV Guidelines:
# • Following rules indicate that a plate should be rejected:
# (a) Does not have at least one letter. This provision shall not apply to plates issued to public officers.
# (b) Has numbers and letters, or any combination thereof, arranged in a format reserved for issuance to specific classes of vehicles other than passenger vehicles.
# (c) Is assigned for issuance to historical motor vehicles.
# (d) Consists of six numbers followed by one letter.
# (e) Is, in the discretion of the commissioner, obscene, lewd, lascivious, derogatory to a particular ethnic or other group, or patently offensive.
# (f) Would lead one to believe that the owner of a particular vehicle is connected with or operating in an official capacity for a governmental organization or function.

# Additional guidelines:
# (i) Are obscene, profane, vulgar, repulsive, depraved, or lewd.
# (ii) Describe or refer to a sexual or intimate body part, area or function.
# (iii) Describe or refer to eliminatory or other bodily functions.
# (iv) Are derogatory, contemptuous, degrading, disrespectful or inflammatory.
# (v) Express, describe, advertise, advocate, promote, encourage, glorify, or condone violence, crime or unlawful conduct.
# (vi) Describe, connote, or refer to illegal drug(s), controlled substance(s) or related paraphernalia.
# (vii) May constitute copyright infringement, or infringement of a trademark, trade name, service mark, or patent.
# (viii) Refer to, suggest, or may appear to refer to or to suggest any governmental or law enforcement purpose, function or entity.
# (ix) Do not have at least one letter, consist of six numbers followed by one letter, or may be misleading or confusing in identifying a plate number (e.g., the substitution of the numeral zero for the letter "O").
# (x) Are reserved for issuance to specific classes of vehicles other than passenger vehicles.

# You must respond with ONLY "ACCEPTED" or "REJECTED". Do not provide any explanation or additional text in your response.
# Output Format:
# [Your decision]\n
# """


    system_prompt = f"""You are an AI assistant for a Department of Motor Vehicles (DMV) tasked with reviewing personalized license plate requests. Your job is to evaluate if a requested plate should be ACCEPTED or REJECTED based on DMV guidelines.

Guidelines for evaluation:
1. The plate must meet character count limits (2-8 characters) and use only allowed characters (A-Z, 0-9, spaces, limited special characters).
Limited special characters and their meanings: '#' - HAND, '$' - HEART, '+' - PLUS sign, '&' - STAR.
2. Prohibited content includes:
   - Obscenity, vulgarity, or offensive language
   - Sexual, excretory, or drug references
   - Violent, discriminatory, or derogatory terms
   - Misleading official terms or confusing formats
   - Copyrighted or trademarked phrases
3. Consider if the plate has a positive or neutral connotation, such as:
   - Creativity, humor, or cleverness
   - Achievements, cultural pride, or inclusivity
   - Positive emotions, hobbies, or playful self-expression
4. The plate must not conflict with reserved formats or special classes of vehicles.
5. It must not be misleading, confusing, or imply governmental affiliation.

You must respond with ONLY "ACCEPTED" or "REJECTED". Do not provide any explanation or additional text in your response.
Output Format:
[Your decision]\n
"""

    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Classify this licence plate: {plate}\n"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt").to(model.device)

    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=5,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[input_ids.shape[-1]:]
    # print(tokenizer.decode(response, skip_special_tokens=True))

    decision = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # decision = decision.lower().replace(system_prompt.lower(), '').strip()
    decision = extract_assistant_response(decision, system_prompt)
    decision = 'ACCEPTED' if 'ACCEPTED' in decision.upper() else 'REJECTED'
    return decision

def process_csv(input_file, output_file, output_dir):
    print(f"Reading input CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"CSV file loaded successfully. Total records: {len(df)}")

    # Load red-guide.csv
    red_guide = pd.read_csv("/home/vivora/data/red-guide.csv")
    red_guide_plates = set(red_guide['plate'].str.upper())

    # Prepare labels
    df['label'] = (df['status'] == 'Y').astype(int) # Why?

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # # For testing purposes
    # test_df = df.sample(n=35, random_state=42)
    # print(test_df)

    decisions = []
    for plate in tqdm(test_df['plate'], desc="Processing plates"):
        if plate.upper() in red_guide_plates:
            decisions.append("REJECTED")
        else:
            decision = generate_decision(plate)
            decisions.append(decision)

    # test_df['model_decision'] = decisions
    test_df['predicted'] = [('Y' if decision == 'ACCEPTED' else 'N') for decision in decisions]

    # Calculate metrics
    y_true = test_df['status']
    y_pred = test_df['predicted']

    print(y_true)
    print(y_pred)
    
    classification_rep = classification_report(y_true, y_pred, target_names=['Rejected', 'Accepted'])
    confusion_mat = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score((y_true == 'Y').astype(int), (y_pred == 'Y').astype(int))
    f1 = f1_score(y_true, y_pred, average='binary', pos_label='Y')

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    test_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save metrics
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_rep)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_mat))
        f.write(f"\nAUC Score: {auc_score:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

# Example usage
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('/home/vivora/outputs', 'llama_CA_classification_' + timestamp)
input_csv = "/home/vivora/data/cali.csv"  # Input CSV file path
output_csv = os.path.join(output_dir,"vanity_plates_llama_decisions.csv")  # Output CSV file path
process_csv(input_csv, output_csv, output_dir)
