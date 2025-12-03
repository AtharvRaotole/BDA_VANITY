import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Load CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# # Calculate class weights
# def compute_class_weights(labels):
#     class_counts = Counter(labels)
#     total = sum(class_counts.values())
#     return {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
def compute_class_weights(labels):
    # Get the class weights as a dictionary
    class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    # Convert the array into a dictionary with class labels as keys
    return {0: class_weights_array[0], 1: class_weights_array[1]}


# Custom dataset for BERT tokenization
class PlateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }

# Extract BERT embeddings
def extract_embeddings(data_loader, model, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)

# Main pipeline
def main_pipeline(file_path):
    # Load data
    data = load_data(file_path)

    data['is_valid'] = data['plate'].apply(validate_license_plate)
    data = data[data['is_valid'] == 1].drop(columns=['is_valid'])

    # Prepare labels
    data['label'] = (data['status'] == 'Y').astype(int)

    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare datasets
    train_dataset = PlateDataset(train_data['plate'].tolist(), train_data['label'].tolist(), tokenizer, max_length=32)
    val_dataset = PlateDataset(val_data['plate'].tolist(), val_data['label'].tolist(), tokenizer, max_length=32)
    test_dataset = PlateDataset(test_data['plate'].tolist(), test_data['label'].tolist(), tokenizer, max_length=32)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Extract embeddings
    train_embeddings = extract_embeddings(train_loader, model, device)
    val_embeddings = extract_embeddings(val_loader, model, device)
    test_embeddings = extract_embeddings(test_loader, model, device)

    train_labels = train_data['label'].values
    val_labels = val_data['label'].values
    test_labels = test_data['label'].values

    # Compute class weights
    class_weights = compute_class_weights(train_labels)

    # Train and evaluate classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=class_weights),
        # 'SVM': SVC(kernel='linear', class_weight=class_weights, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight=class_weights),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=class_weights[1])  # scale_pos_weight for class imbalance
    }


    for name, clf in classifiers.items():
        clf.fit(train_embeddings, train_labels)

        # Validate and test the model
        val_preds = clf.predict(val_embeddings)
        test_preds = clf.predict(test_embeddings)

        print(f"\n{name}:")
        print("-" * 40)

        # Validation Metrics
        print("Validation Metrics:")
        print(classification_report(val_labels, val_preds, target_names=['Rejected', 'Accepted']))
        print(confusion_matrix(val_labels, val_preds))

        # F1 Score (Validation)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"F1 Score (Validation): {val_f1:.4f}")


        # Test Metrics
        print("Test Metrics:")
        print(classification_report(test_labels, test_preds, target_names=['Rejected', 'Accepted']))
        print(confusion_matrix(test_labels, test_preds))

        # F1 Score (Test)
        test_f1 = f1_score(test_labels, test_preds)
        print(f"F1 Score (Test): {test_f1:.4f}")

# Run pipeline
file_path = "cali_v2.csv"
main_pipeline(file_path)

# Load CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Extract Sentence-BERT embeddings
def extract_sbert_embeddings(model, texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

    # Main pipeline
def main_pipeline(file_path):
    # Load data
    data = load_data(file_path)

    data['is_valid'] = data['plate'].apply(validate_license_plate)
    data = data[data['is_valid'] == 1].drop(columns=['is_valid'])

    # Prepare labels
    data['label'] = (data['status'] == 'Y').astype(int)

    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other Sentence-BERT variants

    # Extract embeddings
    train_embeddings = extract_sbert_embeddings(model, train_data['plate'].tolist())
    val_embeddings = extract_sbert_embeddings(model, val_data['plate'].tolist())
    test_embeddings = extract_sbert_embeddings(model, test_data['plate'].tolist())

    train_labels = train_data['label'].values
    val_labels = val_data['label'].values
    test_labels = test_data['label'].values

    # Compute class weights
    class_weights = compute_class_weights(train_labels)

    # Train and evaluate classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=class_weights),
        # 'SVM': SVC(kernel='linear', class_weight=class_weights, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight=class_weights),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=class_weights[1])  # scale_pos_weight for class imbalance
    }

    for name, clf in classifiers.items():
        clf.fit(train_embeddings, train_labels)

        # Validate and test the model
        val_preds = clf.predict(val_embeddings)
        test_preds = clf.predict(test_embeddings)

        print(f"\n{name}:")
        print("-" * 40)

        # Validation Metrics
        print("Validation Metrics:")
        print(classification_report(val_labels, val_preds, target_names=['Rejected', 'Accepted']))
        print(confusion_matrix(val_labels, val_preds))

        # F1 Score (Validation)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"F1 Score (Validation): {val_f1:.4f}")


        # Test Metrics
        print("Test Metrics:")
        print(classification_report(test_labels, test_preds, target_names=['Rejected', 'Accepted']))
        print(confusion_matrix(test_labels, test_preds))

        # F1 Score (Test)
        test_f1 = f1_score(test_labels, test_preds)
        print(f"F1 Score (Test): {test_f1:.4f}")

# Run pipeline
file_path = "t5_v2_ny_gt_rc.csv"
main_pipeline(file_path)