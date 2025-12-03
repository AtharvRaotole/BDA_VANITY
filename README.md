# Vanity Plate Interpretation Project

A deep learning project for interpreting and understanding vanity license plates using transformer-based models (T5 and BERT).

## 📋 Project Overview

This project develops machine learning models to decode the intended meaning behind vanity license plates. The system learns from DMV reviewer comments and slang/abbreviation dictionaries to understand the creative spellings and abbreviations commonly used in personalized plates.

### Key Features

- **T5-based Translation**: Fine-tuned T5-large model for seq-to-seq plate interpretation
- **BERT Embeddings**: Slang-aware embeddings for understanding informal language
- **Multiple Training Approaches**: 
  - Direct plate-to-meaning training on DMV data
  - Transfer learning from slang/abbreviation datasets
  - Fusion models combining T5 and BERT representations

## 📁 Project Structure

```
dsf_project/
├── data/                       # Processed datasets
│   ├── cali.csv               # California DMV vanity plate applications
│   ├── cali_v2.csv            # Enhanced version with additional processing
│   ├── cali_v2_llama_rc.csv   # Plates with LLaMA-generated meanings
│   ├── cali_slang_abb.csv     # Combined plate + slang data
│   ├── slang_abb.csv          # Merged slang and abbreviations
│   └── urbandict-word-defs.csv # Urban Dictionary definitions
│
├── data_dirs/                  # Raw data sources
│   ├── ca-license-plates/     # California license plate data
│   ├── license-plates/        # NY license plate data (for reference)
│   └── slang/                 # Slang and abbreviation source files
│
├── scripts/                    # Training and inference scripts
│   ├── config.py              # Centralized configuration
│   ├── t5.py                  # T5 vanity plate training
│   ├── t5_v2.py               # T5 training (v2 dataset)
│   ├── t5_v2_llama_rc.py      # T5 training (LLaMA-enhanced)
│   ├── t5_ft_slang.py         # T5 slang fine-tuning
│   ├── t5_ft_slang_abb.py     # T5 slang+abbreviation fine-tuning
│   ├── finetune_bert_slang.py # BERT slang embedding training
│   ├── t5_inference.py        # T5 model inference
│   └── *.job                  # SLURM job scripts for HPC
│
├── model_checkpoints/          # Saved model weights
├── outputs/                    # Prediction outputs and results
├── logs/                       # Training logs
├── slurm_logs/                # HPC job logs
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dsf_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

**Train T5 on vanity plates:**
```bash
cd scripts
python t5.py
```

**Train slang-aware BERT embeddings:**
```bash
cd scripts
python finetune_bert_slang.py
```

**Run inference on trained model:**
```bash
cd scripts
python t5_inference.py
```

## 📊 Datasets

### California DMV Data (`cali.csv`)
- Real vanity plate applications from California DMV
- Contains: plate configuration, review reason code, customer meaning, reviewer comments, approval status
- Used for training the primary plate interpretation model

### Urban Dictionary (`urbandict-word-defs.csv`)
- Slang words and their definitions
- Used for pre-training models to understand informal language

### Slang & Abbreviations (`slang_abb.csv`)
- Combined dataset of internet slang and common abbreviations
- Helps model understand shortened forms common in plates

## 🧠 Models

### 1. T5 Vanity Plate Interpreter
- **Base Model**: T5-large (770M parameters)
- **Task**: Seq-to-seq translation from plate → meaning
- **Input**: "Translate vanity plate: {PLATE}"
- **Output**: Interpreted meaning

### 2. BERT Slang Embeddings
- **Base Model**: BERT-base-uncased
- **Task**: Generate meaningful embeddings for slang terms
- **Training**: Contrastive learning on word-definition pairs

### 3. T5 + BERT Fusion (Experimental)
- Combines T5's generation with BERT's semantic understanding
- Projects BERT embeddings to augment T5 encoder representations

## 📈 Training

### Hyperparameters (T5)
| Parameter | Value |
|-----------|-------|
| Model | t5-large |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Max Epochs | 20 |
| Early Stopping Patience | 5 |
| Max Input Length | 32 |
| Max Target Length | 100-128 |

### Training on HPC (SLURM)
```bash
sbatch scripts/t5.job
```

## 📝 Configuration

All hyperparameters and paths are centralized in `scripts/config.py`. Modify this file to change:
- Model parameters
- Data paths
- Training settings
- Device configuration

## 🔧 Enhancement Suggestions

Since this is an older project, here are recommended improvements:

### High Priority
1. **Add Evaluation Metrics**: Implement BLEU, ROUGE, and semantic similarity scores
2. **Upgrade Models**: Try T5-v1.1, Flan-T5, or newer architectures
3. **Data Augmentation**: Generate more training examples using LLMs
4. **Cross-validation**: Implement k-fold validation for robust evaluation

### Medium Priority
5. **Add CLI Interface**: argparse-based command line for flexible training
6. **Implement Logging with TensorBoard/W&B**: Better experiment tracking
7. **Add Unit Tests**: Test data loading, model inference
8. **Docker Container**: Containerize for reproducibility

### Future Directions
9. **Multi-task Learning**: Train on plates + slang simultaneously
10. **Character-level Models**: Better handle creative spellings
11. **Retrieval-Augmented Generation**: Use similarity search for rare terms
12. **API Deployment**: Flask/FastAPI endpoint for inference

## 📄 License

This project is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📚 References

- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [California DMV Personalized Plate Data](https://data.ca.gov/)

---

*Last Updated: December 2025*

