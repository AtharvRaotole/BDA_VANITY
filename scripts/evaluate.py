"""
Evaluation Module for Vanity Plate Interpretation
=================================================
Comprehensive evaluation with multiple metrics including LLM-as-a-Judge.

Metrics Included:
1. Traditional NLP Metrics:
   - BLEU Score
   - ROUGE-1, ROUGE-2, ROUGE-L
   - Cosine Similarity (using sentence embeddings)
   
2. LLM-as-a-Judge:
   - Uses a larger model to score outputs
   - Evaluates: Accuracy, Completeness, Slang Detection, Tone
   - Returns scores 1-5 with reasoning

Supported LLM Backends:
- Azure OpenAI (GPT-4o, GPT-4o-mini)
- OpenAI API (GPT-4, GPT-3.5-turbo)
- Ollama (local Llama 3, Mistral, etc.)

Usage:
    # Traditional metrics only
    python evaluate.py --predictions outputs/predictions.csv
    
    # With Azure OpenAI LLM-as-a-Judge
    python evaluate.py --predictions outputs/predictions.csv --llm-judge --azure \
        --azure-endpoint YOUR_ENDPOINT --azure-key YOUR_KEY --azure-deployment YOUR_DEPLOYMENT
    
    # With OpenAI
    python evaluate.py --predictions outputs/predictions.csv --llm-judge --openai-key YOUR_KEY

Author: AMS 560
Last Updated: December 2025
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging
import os

# Traditional metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: Install nltk and rouge_score for traditional metrics")

# Sentence embeddings for cosine similarity
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: Install sentence-transformers for cosine similarity")

# OpenAI
try:
    import openai
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Requests for Ollama
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "evaluations"
EVAL_DIR.mkdir(exist_ok=True)


# =============================================================================
# LLM-AS-A-JUDGE PROMPT
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for vanity license plate interpretations.

Your task is to evaluate how well a predicted interpretation matches the true meaning of a vanity plate.

Vanity plates use creative spellings, abbreviations, numbers-as-letters (like 4=for, 2=to, 8=ate), 
and slang to convey messages in limited characters.

Evaluate on these criteria (1-5 scale each):

1. **Accuracy** (1-5): Does the prediction capture the correct meaning?
   - 5: Perfect match or semantically equivalent
   - 3: Partially correct, captures some meaning
   - 1: Completely wrong or unrelated

2. **Completeness** (1-5): Does it capture the full intended message?
   - 5: All components of the meaning are present
   - 3: Main idea present but missing details
   - 1: Missing major parts of the meaning

3. **Slang Detection** (1-5): Does it correctly identify slang/abbreviations?
   - 5: All slang/number-substitutions correctly decoded
   - 3: Some slang decoded correctly
   - 1: Failed to recognize slang patterns

4. **Overall** (1-5): Overall quality of the interpretation
   - 5: Would pass as human-quality interpretation
   - 3: Acceptable but has issues
   - 1: Unacceptable interpretation

Respond ONLY with valid JSON in this exact format:
{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "slang_detection": <1-5>,
    "overall": <1-5>,
    "reasoning": "<brief explanation>"
}"""

JUDGE_USER_PROMPT = """Evaluate this vanity plate interpretation:

**Plate**: {plate}
**Predicted Meaning**: {predicted}
**True Meaning**: {reference}

Provide your evaluation as JSON:"""


# =============================================================================
# TRADITIONAL METRICS
# =============================================================================

def compute_bleu(prediction: str, reference: str) -> float:
    """Compute BLEU score between prediction and reference."""
    if not NLTK_AVAILABLE:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    pred_tokens = prediction.lower().split()
    ref_tokens = [reference.lower().split()]
    
    try:
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    except:
        score = 0.0
    
    return score


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    if not NLTK_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }


class CosineSimilarityScorer:
    """Compute cosine similarity using sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if EMBEDDINGS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            self.model = None
    
    def score(self, prediction: str, reference: str) -> float:
        if self.model is None:
            return 0.0
        
        embeddings = self.model.encode([prediction, reference], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), 
            embeddings[1].unsqueeze(0)
        )
        return similarity.item()
    
    def score_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        if self.model is None:
            return [0.0] * len(predictions)
        
        pred_embeddings = self.model.encode(predictions, convert_to_tensor=True)
        ref_embeddings = self.model.encode(references, convert_to_tensor=True)
        
        similarities = torch.nn.functional.cosine_similarity(
            pred_embeddings, ref_embeddings
        )
        return similarities.tolist()


# =============================================================================
# LLM-AS-A-JUDGE
# =============================================================================

@dataclass
class JudgeScore:
    """Container for LLM judge scores."""
    accuracy: int
    completeness: int
    slang_detection: int
    overall: int
    reasoning: str
    
    @property
    def average(self) -> float:
        return (self.accuracy + self.completeness + self.slang_detection + self.overall) / 4


class LLMJudge:
    """LLM-as-a-Judge evaluator supporting multiple backends."""
    
    def __init__(
        self,
        backend: str = "azure",
        model: str = "gpt-4o-mini",
        # Azure settings
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = "2024-12-01-preview",
        # OpenAI settings
        openai_api_key: Optional[str] = None,
        # Ollama settings
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize the LLM Judge.
        
        Args:
            backend: "azure", "openai", or "ollama"
            model: Model name for OpenAI/Ollama
            azure_*: Azure OpenAI settings
            openai_api_key: OpenAI API key
            ollama_url: URL for Ollama server
        """
        self.backend = backend
        self.model = model
        self.ollama_url = ollama_url
        self.azure_deployment = azure_deployment or model
        
        if backend == "azure":
            if not azure_endpoint or not azure_key:
                raise ValueError("Azure endpoint and key required")
            self.client = AzureOpenAI(
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
            )
        elif backend == "openai":
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = OpenAI(api_key=api_key)
    
    def _call_azure(self, plate: str, predicted: str, reference: str) -> Dict:
        """Call Azure OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    plate=plate, predicted=predicted, reference=reference
                )}
            ],
            temperature=0.1,
            max_completion_tokens=500  # Use max_completion_tokens for newer Azure models
        )
        content = response.choices[0].message.content
        return self._parse_response(content)
    
    def _call_openai(self, plate: str, predicted: str, reference: str) -> Dict:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    plate=plate, predicted=predicted, reference=reference
                )}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content
        return self._parse_response(content)
    
    def _call_ollama(self, plate: str, predicted: str, reference: str) -> Dict:
        """Call local Ollama API."""
        prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{JUDGE_USER_PROMPT.format(plate=plate, predicted=predicted, reference=reference)}"
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
        )
        content = response.json()["response"]
        return self._parse_response(content)
    
    def _parse_response(self, content: str) -> Dict:
        """Parse JSON response from LLM."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {
            "accuracy": 3,
            "completeness": 3,
            "slang_detection": 3,
            "overall": 3,
            "reasoning": "Failed to parse LLM response"
        }
    
    def evaluate(self, plate: str, predicted: str, reference: str) -> JudgeScore:
        """Evaluate a single prediction."""
        if self.backend == "azure":
            result = self._call_azure(plate, predicted, reference)
        elif self.backend == "openai":
            result = self._call_openai(plate, predicted, reference)
        elif self.backend == "ollama":
            result = self._call_ollama(plate, predicted, reference)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return JudgeScore(
            accuracy=result.get("accuracy", 3),
            completeness=result.get("completeness", 3),
            slang_detection=result.get("slang_detection", 3),
            overall=result.get("overall", 3),
            reasoning=result.get("reasoning", "")
        )
    
    def evaluate_batch(
        self, 
        plates: List[str], 
        predictions: List[str], 
        references: List[str],
        max_samples: Optional[int] = None
    ) -> List[JudgeScore]:
        """Evaluate a batch of predictions."""
        if max_samples:
            plates = plates[:max_samples]
            predictions = predictions[:max_samples]
            references = references[:max_samples]
        
        scores = []
        for plate, pred, ref in tqdm(zip(plates, predictions, references), 
                                      total=len(plates), 
                                      desc="LLM Judge Evaluation"):
            try:
                score = self.evaluate(plate, pred, ref)
                scores.append(score)
            except Exception as e:
                logging.warning(f"Error evaluating {plate}: {e}")
                scores.append(JudgeScore(3, 3, 3, 3, f"Error: {str(e)}"))
        
        return scores


# =============================================================================
# FULL EVALUATION PIPELINE
# =============================================================================

class VanityPlateEvaluator:
    """Complete evaluation pipeline for vanity plate interpretation."""
    
    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_backend: str = "azure",
        llm_model: str = "gpt-4o-mini",
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434"
    ):
        # Initialize cosine similarity scorer
        self.cosine_scorer = CosineSimilarityScorer() if EMBEDDINGS_AVAILABLE else None
        
        # Initialize LLM judge if requested
        self.llm_judge = None
        if use_llm_judge:
            try:
                self.llm_judge = LLMJudge(
                    backend=llm_backend,
                    model=llm_model,
                    azure_endpoint=azure_endpoint,
                    azure_key=azure_key,
                    azure_deployment=azure_deployment,
                    openai_api_key=openai_api_key,
                    ollama_url=ollama_url
                )
            except Exception as e:
                logging.warning(f"Failed to initialize LLM judge: {e}")
    
    def evaluate_dataframe(
        self, 
        df: pd.DataFrame,
        plate_col: str = "Plate",
        pred_col: str = "Predicted",
        ref_col: str = "Reference",
        llm_judge_samples: Optional[int] = 100
    ) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate a full predictions DataFrame."""
        plates = df[plate_col].tolist()
        predictions = df[pred_col].tolist()
        references = df[ref_col].tolist()
        
        results = []
        
        # Compute traditional metrics
        logging.info("Computing traditional metrics...")
        for plate, pred, ref in tqdm(zip(plates, predictions, references), 
                                      total=len(plates), desc="Traditional Metrics"):
            row = {"plate": plate, "prediction": pred, "reference": ref}
            
            if NLTK_AVAILABLE:
                row["bleu"] = compute_bleu(str(pred), str(ref))
                rouge = compute_rouge(str(pred), str(ref))
                row.update({
                    "rouge1_f": rouge["rouge1"],
                    "rouge2_f": rouge["rouge2"],
                    "rougeL_f": rouge["rougeL"]
                })
            
            results.append(row)
        
        # Batch cosine similarity
        if self.cosine_scorer:
            logging.info("Computing cosine similarities...")
            cos_sims = self.cosine_scorer.score_batch(
                [str(p) for p in predictions], 
                [str(r) for r in references]
            )
            for i, sim in enumerate(cos_sims):
                results[i]["cosine_similarity"] = sim
        
        # LLM judge
        if self.llm_judge and llm_judge_samples:
            logging.info(f"Running LLM Judge on {llm_judge_samples} samples...")
            judge_scores = self.llm_judge.evaluate_batch(
                plates[:llm_judge_samples],
                [str(p) for p in predictions[:llm_judge_samples]],
                [str(r) for r in references[:llm_judge_samples]],
                max_samples=llm_judge_samples
            )
            
            for i, score in enumerate(judge_scores):
                results[i].update({
                    "judge_accuracy": score.accuracy,
                    "judge_completeness": score.completeness,
                    "judge_slang_detection": score.slang_detection,
                    "judge_overall": score.overall,
                    "judge_average": score.average,
                    "judge_reasoning": score.reasoning
                })
        
        results_df = pd.DataFrame(results)
        summary = self._compute_summary(results_df)
        
        return results_df, summary
    
    def _compute_summary(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics."""
        summary = {}
        
        metric_cols = ["bleu", "rouge1_f", "rouge2_f", "rougeL_f", "cosine_similarity"]
        for col in metric_cols:
            if col in df.columns:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
        
        judge_cols = ["judge_accuracy", "judge_completeness", "judge_slang_detection", 
                      "judge_overall", "judge_average"]
        for col in judge_cols:
            if col in df.columns:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
        
        return summary


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vanity plate interpretation predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Traditional metrics only
    python evaluate.py --predictions outputs/predictions.csv
    
    # With Azure OpenAI LLM-as-a-Judge
    python evaluate.py --predictions outputs/predictions.csv --llm-judge --azure \\
        --azure-endpoint https://your-endpoint.openai.azure.com/ \\
        --azure-key YOUR_KEY \\
        --azure-deployment gpt-4o-mini
    
    # With OpenAI
    python evaluate.py --predictions outputs/predictions.csv --llm-judge \\
        --openai-key sk-...
    
    # With local Ollama
    python evaluate.py --predictions outputs/predictions.csv --llm-judge --ollama \\
        --model llama3
        """
    )
    
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--llm-samples", type=int, default=100)
    
    # Azure OpenAI
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI")
    parser.add_argument("--azure-endpoint", type=str)
    parser.add_argument("--azure-key", type=str)
    parser.add_argument("--azure-deployment", type=str, default="gpt-4o-mini")
    
    # OpenAI
    parser.add_argument("--openai-key", type=str)
    
    # Ollama
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Determine backend
    if args.azure:
        backend = "azure"
    elif args.ollama:
        backend = "ollama"
    else:
        backend = "openai"
    
    # Load predictions
    logging.info(f"Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)
    logging.info(f"Loaded {len(df)} predictions")
    
    # Initialize evaluator
    evaluator = VanityPlateEvaluator(
        use_llm_judge=args.llm_judge,
        llm_backend=backend,
        llm_model=args.model,
        azure_endpoint=args.azure_endpoint,
        azure_key=args.azure_key,
        azure_deployment=args.azure_deployment,
        openai_api_key=args.openai_key,
        ollama_url=args.ollama_url
    )
    
    # Run evaluation
    results_df, summary = evaluator.evaluate_dataframe(
        df, llm_judge_samples=args.llm_samples if args.llm_judge else None
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n📊 Traditional Metrics:")
    for key in ["bleu_mean", "rouge1_f_mean", "rouge2_f_mean", "rougeL_f_mean", "cosine_similarity_mean"]:
        if key in summary:
            print(f"  {key.replace('_mean', '').upper():20s}: {summary[key]:.4f}")
    
    if args.llm_judge:
        print("\n🤖 LLM-as-a-Judge Scores (1-5 scale):")
        for key in ["judge_accuracy_mean", "judge_completeness_mean", 
                    "judge_slang_detection_mean", "judge_overall_mean"]:
            if key in summary:
                label = key.replace("_mean", "").replace("judge_", "").replace("_", " ").title()
                print(f"  {label:20s}: {summary[key]:.2f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        pred_path = Path(args.predictions)
        output_path = EVAL_DIR / f"eval_{pred_path.stem}.csv"
    
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
