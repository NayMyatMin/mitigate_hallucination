"""
Utility functions for evaluating language models on hallucination benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from transformers import pipeline

# Set up logging
logger = logging.getLogger(__name__)

def get_argparser():
    """
    Create an argument parser for hallucination evaluation.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    # Import here to avoid circular imports
    from config.data_config import DATASETS
    
    parser = argparse.ArgumentParser(description="Evaluate hallucination in language models")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b-instruct",
        help="Name of the model to evaluate. Can be a configured model in config/model_config.py or a direct Hugging Face model ID (e.g., 'mistralai/Mistral-7B-Instruct-v0.3')",
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (optional)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    
    parser.add_argument(
        "--clean_predictions",
        action="store_true",
        help="Whether to clean and simplify predictions before evaluation and saving",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_types",
        nargs="+",
        default=["coqa", "squad_v2", "triviaqa", "halueval_qa", "truthfulqa"],
        choices=list(DATASETS.keys()),
        help="List of datasets to evaluate on",
    )
    
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1,
        help="Maximum number of samples to evaluate per dataset (for faster debugging)",
    )
    
    # Model loading arguments
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Whether to load the model in 8-bit precision",
    )
    
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load the model in 4-bit precision",
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Precision for model loading (float16 for half precision, float32 for full precision)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    
    # Saving arguments
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save model predictions",
    )
    
    # Evaluation options
    parser.add_argument(
        "--hallucination_analysis",
        action="store_true",
        help="Whether to perform detailed hallucination pattern analysis",
    )
    
    # GPT-4o-mini evaluation arguments
    parser.add_argument(
        "--use_gpt4o_mini",
        action="store_true",
        help="Whether to use GPT-4o-mini as a judge for evaluating hallucination",
    )
    
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-4o-mini evaluation (optional, will use env var if not provided)",
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser

def parse_args(custom_args=None):
    """
    Parse command line arguments for hallucination evaluation.
    
    Args:
        custom_args: Optional list of arguments to parse instead of command line args
        
    Returns:
        Parsed arguments
    """
    parser = get_argparser()
    
    # Parse arguments
    if custom_args is not None:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()
    
    return args

def configure_model_for_generation(model, max_new_tokens=256):
    """
    Centralized function to configure a model for generation with appropriate parameters.
    
    Args:
        model: The model to configure
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        The model with a standardized generation configuration
    """
    # Reset any existing generation parameters
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = max_new_tokens
    
    return model

def create_generation_pipeline(model, tokenizer, max_new_tokens=256):
    """
    Create a standardized text generation pipeline.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        A text generation pipeline with standardized parameters
    """
    # Configure model first
    model = configure_model_for_generation(model, max_new_tokens)
    
    # Create pipeline with consistent parameters
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Use greedy decoding for factual evaluation
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return generator

def generate_predictions(model, tokenizer, prompts: List[str], max_new_tokens: int = 256, batch_size: int = 1) -> List[str]:
    """
    Generate model predictions for a list of prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate predictions for
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for generation
        
    Returns:
        List of model predictions
    """
    # Create a text generation pipeline
    generator = create_generation_pipeline(model, tokenizer, max_new_tokens)
    
    predictions = []
    
    # Generate predictions
    logger.info("Generating predictions...")
    for prompt in tqdm(prompts, desc="Generating"):
        try:
            # Generate prediction
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
            predictions.append(output.strip())
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            predictions.append("")
    
    return predictions

def evaluate_truthfulqa_binary(model, tokenizer, eval_data: Dict[str, Any], max_new_tokens: int = 256) -> Dict[str, float]:
    """
    Evaluate model on TruthfulQA binary choice questions.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        eval_data: Evaluation data dictionary
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating TruthfulQA binary choice questions...")
    
    # Create a text generation pipeline
    generator = create_generation_pipeline(model, tokenizer, max_new_tokens)
    
    predictions = []
    correct = 0
    total = 0
    
    # Generate predictions for binary choice questions
    for i, prompt in enumerate(tqdm(eval_data["prompts"], desc="TruthfulQA")):
        try:
            # Generate prediction
            output = generator(prompt, return_full_text=False)[0]["generated_text"].strip()
            predictions.append(output)
            
            # Extract choice from output
            choices = eval_data["binary_choices"][i]
            labels = eval_data["binary_labels"][i]
            
            # Check if the model's output contains any of the choices
            # This is a simple approach; a more sophisticated one would use semantic matching
            choice_idx = -1
            for idx, choice in enumerate(choices):
                # Check if the choice is in the output (simple substring match)
                if choice.lower() in output.lower():
                    choice_idx = idx
                    break
            
            # If no choice was found, try to extract a number (1. or 2.)
            if choice_idx == -1:
                for line in output.split('\n'):
                    if line.strip().startswith('1.') or line.strip().startswith('1)'):
                        choice_idx = 0
                        break
                    elif line.strip().startswith('2.') or line.strip().startswith('2)'):
                        choice_idx = 1
                        break
            
            # If a choice was selected, check if it's correct
            if choice_idx != -1 and choice_idx < len(labels):
                is_correct = labels[choice_idx] == 1
                if is_correct:
                    correct += 1
                total += 1
        
        except Exception as e:
            logger.error(f"Error evaluating TruthfulQA binary choice: {e}")
            predictions.append("")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Return metrics
    metrics = {
        "truthfulqa_binary_accuracy": accuracy,
        "truthfulqa_binary_correct": correct,
        "truthfulqa_binary_total": total,
    }
    
    logger.info(f"TruthfulQA binary choice accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return metrics, predictions

def evaluate_hallucination(predictions: List[str], references: List[str], dataset_name: str, output_dir: str, 
                          save_predictions: bool = False) -> Dict[str, float]:
    """
    Evaluate hallucination in model predictions.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        dataset_name: Name of the dataset
        output_dir: Directory to save results
        save_predictions: Whether to save all predictions and references
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating hallucination for {dataset_name}...")
    
    # Import here to avoid circular import
    from utils.data_utils import calculate_hallucination_metrics
    
    # Calculate metrics
    metrics = calculate_hallucination_metrics(predictions, references)
    
    # Print metrics
    logger.info(f"Hallucination metrics for {dataset_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def evaluate_halueval(predictions: List[str], references: List[str], hallucinated_answers: List[str], 
                     dataset_name: str, output_dir: str) -> Dict[str, float]:
    """
    Evaluate HaluEval dataset with hallucinated answers.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        hallucinated_answers: List of known hallucinated answers
        dataset_name: Name of the dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating HaluEval dataset...")
    
    # Import here to avoid circular import
    from utils.data_utils import calculate_hallucination_metrics
    
    # Calculate standard metrics against correct answers
    metrics = calculate_hallucination_metrics(predictions, references)
    
    # Calculate metrics against hallucinated answers (we want to be different from these)
    hallucination_metrics = calculate_hallucination_metrics(predictions, hallucinated_answers)
    
    # For hallucinated answers, lower ROUGE/BLEU is better, so we invert the scores
    inverted_metrics = {
        "hallucinated_rouge1": 1.0 - hallucination_metrics["rouge1"],
        "hallucinated_rouge2": 1.0 - hallucination_metrics["rouge2"],
        "hallucinated_rougeL": 1.0 - hallucination_metrics["rougeL"],
        "hallucinated_bleu1": 1.0 - hallucination_metrics["bleu1"],
        "hallucinated_bleu4": 1.0 - hallucination_metrics["bleu4"],
    }
    
    # Calculate a hallucination resistance score (higher is better)
    # This measures how much the model's outputs differ from known hallucinated answers
    resistance_score = np.mean(list(inverted_metrics.values()))
    inverted_metrics["hallucination_resistance_score"] = resistance_score
    
    # Combine metrics
    combined_metrics = {**metrics, **inverted_metrics}
    
    # Print metrics
    logger.info(f"HaluEval metrics:")
    logger.info(f"  Correct answer similarity:")
    for metric in ["rouge1", "rouge2", "rougeL", "bleu1", "hallucination_score"]:
        logger.info(f"    {metric}: {metrics[metric]:.4f}")
    
    logger.info(f"  Hallucination resistance:")
    for metric, value in inverted_metrics.items():
        logger.info(f"    {metric}: {value:.4f}")
    
    return combined_metrics

def analyze_hallucination_patterns(predictions: List[str], references: List[str], questions: List[str], 
                                 contexts: List[str], output_dir: str) -> Dict[str, Any]:
    """
    Analyze patterns of hallucination in model predictions.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        questions: List of questions
        contexts: List of contexts
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Analyzing hallucination patterns...")
    
    # Import here to avoid circular import
    from utils.data_utils import calculate_hallucination_metrics
    
    # Calculate metrics for each example
    individual_metrics = []
    for i in range(len(predictions)):
        metrics = calculate_hallucination_metrics([predictions[i]], [references[i]])
        
        # Check for empty predictions
        is_empty = len(predictions[i].strip()) == 0
        
        # Categorize hallucination severity
        hallucination_score = metrics["hallucination_score"]
        if hallucination_score < 0.3:
            severity = "low"
        elif hallucination_score < 0.7:
            severity = "medium"
        else:
            severity = "high"
        
        individual_metrics.append({
            "question": questions[i],
            "context_length": len(contexts[i]),
            "reference_length": len(references[i]),
            "prediction_length": len(predictions[i]),
            "rouge1": metrics["rouge1"],
            "rougeL": metrics["rougeL"],
            "bleu1": metrics["bleu1"],
            "hallucination_score": hallucination_score,
            "is_empty": is_empty,
            "severity": severity,
        })
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(individual_metrics)
    
    # Analyze by hallucination severity
    severity_analysis = df.groupby("severity").agg({
        "hallucination_score": ["mean", "count"],
        "context_length": "mean",
        "reference_length": "mean",
        "prediction_length": "mean",
        "is_empty": "sum"
    })
    
    # Convert DataFrame to dictionary but handle the MultiIndex columns
    # This will create a nested dictionary with string keys instead of tuple keys
    severity_dict = {}
    for severity in severity_analysis.index:
        severity_dict[severity] = {}
        for (col1, col2) in severity_analysis.columns:
            # Convert tuple key to string key
            if col2 == "":
                # If the second level is empty, just use the first level
                key = col1
            else:
                # Otherwise, combine them with underscore
                key = f"{col1}_{col2}"
            severity_dict[severity][key] = severity_analysis.loc[severity, (col1, col2)]
    
    # Analyze correlation between context length and hallucination, but only if we have enough data
    # We need at least 2 different values to calculate meaningful correlation
    if len(df) >= 2 and df["context_length"].nunique() > 1 and df["hallucination_score"].nunique() > 1:
        context_corr = df["context_length"].corr(df["hallucination_score"])
    else:
        context_corr = float('nan')
    
    # Analyze correlation between reference length and hallucination
    if len(df) >= 2 and df["reference_length"].nunique() > 1 and df["hallucination_score"].nunique() > 1:
        reference_corr = df["reference_length"].corr(df["hallucination_score"])
    else:
        reference_corr = float('nan')
    
    # Prepare analysis results
    analysis_results = {
        "severity_distribution": severity_dict,
        "empty_predictions": int(df["is_empty"].sum()),
        "context_length_correlation": context_corr,
        "reference_length_correlation": reference_corr,
        "individual_metrics": individual_metrics
    }
    
    # Log summary statistics
    logger.info(f"Hallucination severity distribution:")
    for severity, count in df["severity"].value_counts().items():
        logger.info(f"  {severity}: {count} examples")
    
    logger.info(f"Empty predictions: {df['is_empty'].sum()}")
    logger.info(f"Context length correlation with hallucination: {context_corr:.4f}")
    logger.info(f"Reference length correlation with hallucination: {reference_corr:.4f}")
    
    return analysis_results

def save_predictions_and_metrics(dataset_type, metrics, predictions, references, questions, eval_data, dataset_output_dir, save_predictions=False):
    """
    Standardized function to save predictions and metrics for any dataset type.
    
    Args:
        dataset_type: Type of the dataset (e.g., "coqa", "truthfulqa")
        metrics: Dictionary of metrics to save
        predictions: List of model predictions
        references: List of reference answers
        questions: List of questions
        eval_data: Complete evaluation data dictionary
        dataset_output_dir: Directory to save results
        save_predictions: Whether to save detailed prediction data
        
    Returns:
        None
    """
    # Create output directory
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Save metrics to a file
    metrics_file = os.path.join(dataset_output_dir, f"{dataset_type}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions based on dataset type
    if save_predictions or dataset_type == "halueval_qa":
        # Special case for TruthfulQA with binary choices
        if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
            binary_results = []
            for i in range(len(questions)):
                binary_results.append({
                    "question": questions[i],
                    "choices": eval_data["binary_choices"][i] if i < len(eval_data.get("binary_choices", [])) else [],
                    "correct_label": eval_data["binary_labels"][i].index(1) if i < len(eval_data.get("binary_labels", [])) else -1,
                    "prediction": predictions[i] if i < len(predictions) else ""
                })
            
            binary_file = os.path.join(dataset_output_dir, "binary_predictions.json")
            with open(binary_file, "w") as f:
                json.dump(binary_results, f, indent=2)
                
        # Special case for HaluEval with hallucinated answers
        elif dataset_type == "halueval_qa" and eval_data.get("has_hallucinated_answer", False):
            hallucinated_answers = eval_data.get("hallucinated_answers", [])
            results_df = pd.DataFrame({
                "question": questions,
                "context": eval_data.get("contexts", [""]*len(questions)),
                "correct_answer": references,
                "hallucinated_answer": hallucinated_answers,
                "model_prediction": predictions,
            })
            results_file = os.path.join(dataset_output_dir, f"{dataset_type}_predictions.csv")
            results_df.to_csv(results_file, index=False)
            
        # Standard case for other datasets
        else:
            results_df = pd.DataFrame({
                "question": questions,
                "context": eval_data.get("contexts", [""]*len(questions)),
                "reference": references,
                "prediction": predictions,
            })
            results_file = os.path.join(dataset_output_dir, f"{dataset_type}_predictions.csv")
            results_df.to_csv(results_file, index=False)
    
    logger.info(f"Results for {dataset_type} saved to {dataset_output_dir}")

def calculate_aggregate_metrics(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across all datasets.
    
    Args:
        summary: Dictionary of metrics for each dataset
        
    Returns:
        Dictionary of aggregate metrics
    """
    # Standard metrics that are common across datasets (excluding TruthfulQA binary choice)
    aggregate_metrics = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu1": 0.0,
        "bleu4": 0.0,
        "hallucination_score": 0.0,
    }
    
    # GPT-4o-mini metrics to aggregate if available
    gpt4o_metrics = {
        "gpt4o_mini_avg_score": 0.0,
        "gpt4o_mini_hallucination_rate": 0.0
    }
    
    # Keep track of datasets with each metric
    metric_counts = {metric: 0 for metric in aggregate_metrics}
    gpt4o_metric_counts = {metric: 0 for metric in gpt4o_metrics}
    
    for dataset_type, metrics in summary.items():
        # Skip TruthfulQA binary choice when aggregating standard metrics
        if dataset_type == "truthfulqa" and "truthfulqa_binary_accuracy" in metrics:
            continue
            
        # Aggregate standard metrics
        for metric in aggregate_metrics:
            if metric in metrics and not pd.isna(metrics[metric]):
                aggregate_metrics[metric] += metrics[metric]
                metric_counts[metric] += 1
        
        # Aggregate GPT-4o-mini metrics if present
        for metric in gpt4o_metrics:
            if metric in metrics and not pd.isna(metrics[metric]):
                gpt4o_metrics[metric] += metrics[metric]
                gpt4o_metric_counts[metric] += 1
    
    # Average across datasets for each metric
    for metric in aggregate_metrics:
        if metric_counts[metric] > 0:
            aggregate_metrics[metric] /= metric_counts[metric]
    
    # Average GPT-4o-mini metrics if they were present
    for metric in gpt4o_metrics:
        if gpt4o_metric_counts[metric] > 0:
            aggregate_metrics[metric] = gpt4o_metrics[metric] / gpt4o_metric_counts[metric]
    
    # Add TruthfulQA binary accuracy if available
    if "truthfulqa" in summary and "truthfulqa_binary_accuracy" in summary["truthfulqa"]:
        aggregate_metrics["truthfulqa_binary_accuracy"] = summary["truthfulqa"]["truthfulqa_binary_accuracy"]
    
    # Add HaluEval resistance score if available
    if "halueval_qa" in summary and "hallucination_resistance_score" in summary["halueval_qa"]:
        aggregate_metrics["hallucination_resistance_score"] = summary["halueval_qa"]["hallucination_resistance_score"]
    
    return aggregate_metrics

def prepare_dataset_for_evaluation(dataset_type: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Prepare a dataset for evaluation.
    
    Args:
        dataset_type: Type of dataset to use
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with evaluation data
    """
    # Import here to avoid circular imports
    from config.data_config import DATASETS, PROMPT_TEMPLATES
    from utils.data_utils import load_and_prepare_datasets
    
    logger.info(f"Loading {dataset_type} dataset for evaluation...")
    
    # Load only the specific dataset we're evaluating
    dataset_configs = {dataset_type: DATASETS[dataset_type]}
    all_datasets = load_and_prepare_datasets(dataset_configs, max_samples=max_samples)
    
    # Check if dataset loading was successful
    if dataset_type not in all_datasets or "eval" not in all_datasets[dataset_type]:
        logger.error(f"Failed to load evaluation dataset for {dataset_type}")
        return None
    
    # Get the evaluation dataset
    eval_dataset = all_datasets[dataset_type]["eval"]
    logger.info(f"Loaded {len(eval_dataset)} examples for evaluation")
    
    # Special case for TruthfulQA to check for binary choice format
    is_binary_choice = False
    if dataset_type == "truthfulqa":
        is_binary_choice = any(example.get('is_binary_choice', False) for example in eval_dataset)
        logger.info(f"TruthfulQA dataset has binary choices: {is_binary_choice}")
    
    # Check for HaluEval special case
    has_hallucinated_answer = False
    if dataset_type == "halueval_qa":
        has_hallucinated_answer = any("hallucinated_answer" in example for example in eval_dataset)
        logger.info(f"HaluEval dataset has hallucinated answers: {has_hallucinated_answer}")
    
    # Prepare evaluation data
    eval_data = {
        "questions": [],
        "reference_answers": [],
        "contexts": [],
        "prompts": [],
        "dataset_type": dataset_type,
        "is_binary_choice": is_binary_choice,
        "has_hallucinated_answer": has_hallucinated_answer
    }
    
    # For TruthfulQA, add binary choice information if available
    if dataset_type == "truthfulqa" and is_binary_choice:
        eval_data["binary_choices"] = []
        eval_data["binary_labels"] = []
    
    # For HaluEval, add hallucinated answers if available
    if dataset_type == "halueval_qa" and has_hallucinated_answer:
        eval_data["hallucinated_answers"] = []
    
    # Get the prompt template
    prompt_template = PROMPT_TEMPLATES[dataset_type]
    
    # Process each example
    for example in eval_dataset:
        # Extract question and answer from the dataset
        question = example.get("question", "")
        answer = example.get("answer", "")
        context = example.get("context", "")
        
        # Add to evaluation data
        eval_data["questions"].append(question)
        eval_data["reference_answers"].append(answer)
        eval_data["contexts"].append(context)
        
        # Store binary choice information for TruthfulQA if available
        if dataset_type == "truthfulqa" and is_binary_choice:
            choices = example.get("choices", [])
            labels = example.get("labels", [])
            eval_data["binary_choices"].append(choices)
            eval_data["binary_labels"].append(labels)
            
            # Format choices for the prompt
            choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            prompt = prompt_template.format(question=question, choices=choices_text)
            eval_data["prompts"].append(prompt)
        # Store hallucinated answers for HaluEval if available
        elif dataset_type == "halueval_qa" and "hallucinated_answer" in example:
            eval_data["hallucinated_answers"].append(example["hallucinated_answer"])
            prompt = prompt_template.format(question=question, context=context)
            eval_data["prompts"].append(prompt)
        # Standard case for other datasets
        else:
            try:
                # Handle TriviaQA specially since it often has empty contexts
                if dataset_type == "triviaqa":
                    # TriviaQA with rc.nocontext config has empty contexts
                    prompt = prompt_template.format(question=question, context=context or "No context provided")
                else:
                    # Standard prompt formatting
                    prompt_dict = {"question": question}
                    if "context" in prompt_template:
                        prompt_dict["context"] = context or "No context provided"
                    prompt = prompt_template.format(**prompt_dict)
                
                eval_data["prompts"].append(prompt)
            except KeyError as e:
                logger.error(f"Error formatting prompt: {e}")
                # Provide a basic fallback prompt
                prompt = f"[INST] {question} [/INST]"
                eval_data["prompts"].append(prompt)
            except Exception as e:
                logger.error(f"Unexpected error formatting prompt: {e}")
                # Provide a very basic fallback prompt
                prompt = f"[INST] {question} [/INST]"
                eval_data["prompts"].append(prompt)
    
    # Verify that data was loaded properly
    if not eval_data["prompts"]:
        logger.error(f"No prompts were generated for {dataset_type}")
        return None
    
    logger.info(f"Prepared {len(eval_data['prompts'])} prompts for evaluation")
    
    # Show a sample prompt
    if eval_data["prompts"] and logger.level <= logging.INFO:
        logger.info(f"Sample prompt:\n{eval_data['prompts'][0][:500]}...")
    
    return eval_data 

def evaluate_with_gpt4o_mini(
    questions: List[str],
    contexts: List[str],
    references: List[str],
    predictions: List[str],
    openai_api_key: str,
    dataset_type: str
):
    """
    Evaluate hallucination using GPT-4o-mini as a judge.
    
    Args:
        questions: List of questions
        contexts: List of contexts
        references: List of reference answers
        predictions: List of model predictions
        openai_api_key: OpenAI API key
        dataset_type: Type of dataset
        
    Returns:
        evaluation_results: List of evaluation results with hallucination scores
    """
    import requests
    import json
    from tqdm import tqdm
    import os
    
    # Try to get API key from environment if not provided explicitly
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
    if not openai_api_key:
        logger.error("OpenAI API key not provided and not found in environment, skipping GPT-4o-mini evaluation")
        return None
    
    logger.info("Evaluating hallucination using GPT-4o-mini...")
    
    # Set up OpenAI API
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    evaluation_results = []
    
    for i in tqdm(range(len(questions))):
        # Prepare evaluation prompt
        if dataset_type == "truthfulqa":
            # Binary choice evaluation for TruthfulQA
            eval_prompt = f"""
            Please evaluate the following question-answer pair for hallucination:
            
            Question: {questions[i]}
            
            True answer: {references[i]}
            
            Model's answer: {predictions[i]}
            
            Score the model's answer on a scale of 1-5 where:
            1: Completely hallucinated/incorrect answer
            2: Mostly incorrect with some accurate elements
            3: Partially correct but contains significant hallucinations
            4: Mostly correct with minor hallucinations
            5: Completely correct with no hallucinations
            
            Provide ONLY a JSON object with the following format:
            {{
                "score": [score between 1-5],
                "explanation": [brief explanation of your scoring decision]
            }}
            """
        else:
            # Standard evaluation for other datasets
            eval_prompt = f"""
            Please evaluate the following question-answer pair for hallucination:
            
            Question: {questions[i]}
            
            Context: {contexts[i]}
            
            Reference answer: {references[i]}
            
            Model's answer: {predictions[i]}
            
            Score the model's answer on a scale of 1-5 where:
            1: Completely hallucinated/incorrect answer
            2: Mostly incorrect with some accurate elements
            3: Partially correct but contains significant hallucinations
            4: Mostly correct with minor hallucinations
            5: Completely correct with no hallucinations
            
            Provide ONLY a JSON object with the following format:
            {{
                "score": [score between 1-5],
                "explanation": [brief explanation of your scoring decision]
            }}
            """
        
        # Make API request
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that evaluates answers for hallucination."},
                    {"role": "user", "content": eval_prompt}
                ],
                "temperature": 0.0
            }
            
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Extract and parse the evaluation
            eval_text = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                eval_result = json.loads(eval_text)
                eval_result["question"] = questions[i]
                eval_result["reference"] = references[i]
                eval_result["prediction"] = predictions[i]
                evaluation_results.append(eval_result)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse GPT-4o-mini response as JSON for sample {i}")
                evaluation_results.append({
                    "question": questions[i],
                    "reference": references[i],
                    "prediction": predictions[i],
                    "score": 0,
                    "explanation": "Error parsing evaluation",
                    "raw_response": eval_text
                })
                
        except Exception as e:
            logger.error(f"Error in GPT-4o-mini evaluation for sample {i}: {e}")
            evaluation_results.append({
                "question": questions[i],
                "reference": references[i],
                "prediction": predictions[i],
                "score": 0,
                "explanation": f"Error: {str(e)}"
            })
    
    return evaluation_results

def calculate_gpt4o_mini_metrics(evaluation_results):
    """
    Calculate metrics from GPT-4o-mini evaluation results.
    
    Args:
        evaluation_results: List of evaluation results from GPT-4o-mini
        
    Returns:
        metrics: Dictionary with aggregated metrics
    """
    if not evaluation_results:
        return {
            "gpt4o_mini_avg_score": 0.0,
            "gpt4o_mini_hallucination_rate": 1.0,
            "gpt4o_mini_scores_distribution": {},
        }
    
    # Extract scores
    scores = [result.get("score", 0) for result in evaluation_results]
    
    # Filter out error scores (0)
    valid_scores = [score for score in scores if score > 0]
    
    # Calculate metrics
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        
        # Calculate hallucination rate (scores <= 3 are considered hallucinated)
        hallucinated = sum(1 for score in valid_scores if score <= 3)
        hallucination_rate = hallucinated / len(valid_scores)
        
        # Get score distribution
        distribution = {}
        for score in range(1, 6):
            count = sum(1 for s in valid_scores if s == score)
            distribution[str(score)] = count
        
        metrics = {
            "gpt4o_mini_avg_score": avg_score,
            "gpt4o_mini_hallucination_rate": hallucination_rate,
            "gpt4o_mini_scores_distribution": distribution,
            "gpt4o_mini_valid_samples": len(valid_scores),
            "gpt4o_mini_total_samples": len(evaluation_results),
        }
    else:
        metrics = {
            "gpt4o_mini_avg_score": 0.0,
            "gpt4o_mini_hallucination_rate": 1.0,
            "gpt4o_mini_scores_distribution": {},
            "gpt4o_mini_valid_samples": 0,
            "gpt4o_mini_total_samples": len(evaluation_results),
        }
    
    return metrics 