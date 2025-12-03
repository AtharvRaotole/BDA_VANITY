# 🖥️ HPC Cluster Running Instructions

Complete guide for running the Vanity Plate Interpretation project on an HPC cluster with SLURM.

---

## 📋 Prerequisites

- Access to HPC cluster with SLURM
- GPU nodes available (NVIDIA GPU recommended)
- Python 3.9+ available as module or installed

---

## 🚀 Step 1: Setup on Cluster

### 1.1 Clone/Upload Project

```bash
# SSH to cluster
ssh your_username@cluster.edu

# Navigate to your work directory
cd /home/your_username/  # or your scratch space

# Clone or upload the project
# Option A: Clone from git
git clone https://github.com/AtharvRaotole/BDA_VANITY.git

# Option B: Upload via scp (from your Mac)
# scp -r /Users/atharvraotole/Downloads/BDA_VANITY your_username@cluster.edu:/home/your_username/
```

### 1.2 Create Virtual Environment

```bash
cd BDA_VANITY

# Load Python module (cluster-specific)
module load python/3.10  # or python/3.9, check with: module avail python

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🏋️ Step 2: Training

### 2.1 Create Training Job Script

Create file `scripts/train_cluster.job`:

```bash
#!/bin/bash
#SBATCH --job-name=vanity_plate_train
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # Change to your GPU partition name
#SBATCH --cpus-per-task=4

# =============================================================================
# Vanity Plate Training Job
# =============================================================================

echo "Starting job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8  # or appropriate CUDA version

# Navigate to project
cd /home/$USER/BDA_VANITY

# Activate environment
source venv/bin/activate

# Check GPU
echo "GPU Info:"
nvidia-smi

# Run training
# Options:
#   --model google/flan-t5-small   (80M params, fastest)
#   --model google/flan-t5-base    (250M params, recommended)
#   --model google/flan-t5-large   (780M params, best quality)

python scripts/train.py \
    --model google/flan-t5-base \
    --batch_size 8 \
    --epochs 20 \
    --lr 5e-5 \
    --data_file cali.csv

echo "Job finished at $(date)"
```

### 2.2 Submit Training Job

```bash
# Make sure log directory exists
mkdir -p slurm_logs

# Submit job
sbatch scripts/train_cluster.job

# Check job status
squeue -u $USER

# Watch job output (once running)
tail -f slurm_logs/train_*.out
```

---

## 📊 Step 3: Evaluation

### 3.1 Create Evaluation Job Script

Create file `scripts/evaluate_cluster.job`:

```bash
#!/bin/bash
#SBATCH --job-name=vanity_plate_eval
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

# =============================================================================
# Vanity Plate Evaluation Job
# =============================================================================

echo "Starting evaluation on $(hostname) at $(date)"

# Load modules
module load python/3.10

cd /home/$USER/BDA_VANITY
source venv/bin/activate

# Find latest predictions file
PRED_FILE=$(ls -t outputs/predictions_*.csv | head -1)
echo "Evaluating: $PRED_FILE"

# ============================================
# Option 1: Traditional metrics only (no API needed)
# ============================================
python scripts/evaluate.py \
    --predictions "$PRED_FILE"

# ============================================
# Option 2: With Azure OpenAI LLM-as-a-Judge
# ============================================
# Uncomment and fill in your Azure details:
#
# python scripts/evaluate.py \
#     --predictions "$PRED_FILE" \
#     --llm-judge \
#     --azure \
#     --azure-endpoint "https://your-endpoint.cognitiveservices.azure.com/" \
#     --azure-key "YOUR_AZURE_KEY" \
#     --azure-deployment "gpt-4o-mini" \
#     --llm-samples 100

echo "Evaluation finished at $(date)"
```

### 3.2 Submit Evaluation Job

```bash
sbatch scripts/evaluate_cluster.job
```

---

## 🔑 Step 4: Azure OpenAI Setup (for LLM-as-a-Judge)

### 4.1 Check Your Azure Deployment Name

Your Azure endpoint works, but you need the correct **deployment name**. 

Go to: Azure Portal → Your OpenAI Resource → Model Deployments

Find the deployment name (e.g., `gpt-4o-mini`, `gpt-4`, etc.)

### 4.2 Test Azure Connection

```bash
# On cluster, test your Azure connection:
python -c "
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version='2024-12-01-preview',
    azure_endpoint='https://manav-mip8bl3j-eastus2.cognitiveservices.azure.com/',
    api_key='YOUR_KEY',
)

# List available deployments (if supported)
try:
    response = client.chat.completions.create(
        model='YOUR_DEPLOYMENT_NAME',  # <-- Change this!
        messages=[{'role': 'user', 'content': 'Say hi'}],
        max_tokens=5
    )
    print('✅ Connected!')
    print(response.choices[0].message.content)
except Exception as e:
    print(f'❌ Error: {e}')
"
```

---

## 📁 Step 5: Useful Commands

### Job Management

```bash
# Submit job
sbatch script.job

# Check your jobs
squeue -u $USER

# Cancel job
scancel JOB_ID

# Check job details
scontrol show job JOB_ID

# View job output
cat slurm_logs/train_12345.out
```

### GPU Monitoring

```bash
# Check GPU availability
sinfo -p gpu

# Interactive GPU session (for debugging)
srun --partition=gpu --gres=gpu:1 --time=1:00:00 --pty bash
```

### Training Progress

```bash
# Watch training logs
tail -f logs/training_*.txt

# Check latest model checkpoint
ls -la model_checkpoints/
```

---

## 📊 Expected Output

After successful training, you'll have:

```
BDA_VANITY/
├── model_checkpoints/
│   └── best_flan_t5_base_20241202_143022/  # Your trained model
├── outputs/
│   └── predictions_flan_t5_base_20241202_143022.csv
├── logs/
│   └── training_flan_t5_base_20241202_143022.txt
├── evaluations/
│   └── eval_predictions_flan_t5_base_20241202_143022.csv
└── slurm_logs/
    └── train_12345.out
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Submit training | `sbatch scripts/train_cluster.job` |
| Submit evaluation | `sbatch scripts/evaluate_cluster.job` |
| Check jobs | `squeue -u $USER` |
| Cancel job | `scancel JOB_ID` |
| View output | `tail -f slurm_logs/train_*.out` |
| GPU info | `nvidia-smi` |

---

## ⚠️ Troubleshooting

### "Module not found"
```bash
module avail python  # Find available Python versions
module avail cuda    # Find available CUDA versions
```

### "Out of memory"
- Reduce batch size: `--batch_size 2`
- Use smaller model: `--model google/flan-t5-small`

### "Job pending"
```bash
squeue -u $USER  # Check why (RESOURCES, PRIORITY, etc.)
```

---

*Last Updated: December 2025*

