#!/usr/bin/env python3
"""
Evaluation script for measuring hallucination in language models.
"""

import os
import argparse
import logging
from datetime import datetime
import torch
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    pipeline
)
from peft import PeftModel
import click

# Local imports
from config.model_config import MODELS
from config.data_config import DATASETS, PROMPT_TEMPLATES
from utils.data_utils import load_and_prepare_datasets, calculate_hallucination_metrics
from utils.model_utils import load_base_model_and_tokenizer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate hallucination in language models")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b-instruct",
        help="Name of the base model to evaluate (from config)",
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (optional)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    
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
        default=100,
        help="Maximum number of samples to evaluate per dataset (for faster debugging)",
    )
    
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def prepare_dataset_for_evaluation(dataset_type, tokenizer, max_samples=None):
    """
    Prepare a dataset for evaluation.
    
    Args:
        dataset_type: Type of dataset to use
        tokenizer: Tokenizer for the model
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Evaluation dataset
    """
    # Load dataset
    logger.info(f"Loading {dataset_type} dataset for evaluation...")
    
    # Load datasets using our improved data_utils
    from utils.data_utils import load_and_prepare_datasets
    from config.data_config import DATASETS, PROMPT_TEMPLATES
    
    # Only load the specific dataset we're evaluating
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
    if dataset_type == "truthfulqa":
        has_binary_choices = any(example.get('is_binary_choice', False) for example in eval_dataset)
        logger.info(f"TruthfulQA dataset has binary choices: {has_binary_choices}")
    
    # Get dataset config
    dataset_config = DATASETS[dataset_type]
    input_key = dataset_config["input_key"]
    output_key = dataset_config["output_key"]
    context_key = dataset_config.get("context_key", None)
    
    # Prepare evaluation data
    eval_data = {
        "questions": [],
        "reference_answers": [],
        "contexts": [],
        "prompts": [],
        "dataset_type": dataset_type
    }
    
    # For TruthfulQA, add binary choice information if available
    if dataset_type == "truthfulqa":
        eval_data["is_binary_choice"] = any(example.get('is_binary_choice', False) for example in eval_dataset)
        eval_data["binary_choices"] = []
        eval_data["binary_labels"] = []
    
    # Get the prompt template
    prompt_template = PROMPT_TEMPLATES[dataset_type]
    
    # Process each example
    for example in eval_dataset:
        question = example[input_key]
        answer = example[output_key]
        context = example.get(context_key, "") if context_key and context_key in example else ""
        
        # Add to evaluation data
        eval_data["questions"].append(question)
        eval_data["reference_answers"].append(answer)
        eval_data["contexts"].append(context)
        
        # Store binary choice information for TruthfulQA if available
        if dataset_type == "truthfulqa" and example.get('is_binary_choice', False):
            eval_data["binary_choices"].append(example.get("choices", []))
            eval_data["binary_labels"].append(example.get("labels", []))
        
        # Create prompt
        prompt_dict = {"question": question}
        if context:
            prompt_dict["context"] = context
        
        try:
            prompt = prompt_template.format(**prompt_dict)
            eval_data["prompts"].append(prompt)
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            eval_data["prompts"].append("")
    
    # Verify that data was loaded properly
    if not eval_data["prompts"]:
        logger.error(f"No prompts were generated for {dataset_type}")
        return None
    
    logger.info(f"Prepared {len(eval_data['prompts'])} prompts for evaluation")
    
    # Show a sample prompt
    if eval_data["prompts"]:
        logger.info(f"Sample prompt:\n{eval_data['prompts'][0]}")
    
    return eval_data


def generate_predictions(model, tokenizer, prompts, max_new_tokens=256):
    """
    Generate model predictions for a list of prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate predictions for
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        List of model predictions
    """
    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Use greedy decoding for deterministic outputs
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    predictions = []
    
    # Generate predictions in batches
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


def evaluate_hallucination(predictions, references, dataset_name, output_dir):
    """
    Evaluate hallucination in model predictions.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        dataset_name: Name of the dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating hallucination for {dataset_name}...")
    
    # Calculate metrics
    metrics = calculate_hallucination_metrics(predictions, references)
    
    # Print metrics
    logger.info(f"Hallucination metrics for {dataset_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics to a file
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and references to a CSV file
    results_df = pd.DataFrame({
        "reference": references,
        "prediction": predictions,
    })
    results_file = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    results_df.to_csv(results_file, index=False)
    
    return metrics


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.replace("/", "_")
    output_dir = os.path.join(args.output_dir, f"{model_name_short}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA adapter: {args.adapter_path if args.adapter_path else 'None'}")
    logger.info(f"Datasets: {args.dataset_types}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=args.model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load LoRA adapter if specified
    if args.adapter_path:
        logger.info(f"Loading LoRA adapter from {args.adapter_path}...")
        try:
            model = PeftModel.from_pretrained(model, args.adapter_path)
            logger.info("LoRA adapter loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LoRA adapter: {e}")
            logger.info("Proceeding with base model only.")
    
    # Configure model for generation
    model.eval()  # Set model to evaluation mode
    
    # Initialize results summary
    summary = {}
    
    # Evaluate on each specified dataset
    for dataset_type in args.dataset_types:
        try:
            logger.info(f"Evaluating on {dataset_type} dataset...")
            
            # Prepare dataset
            eval_data = prepare_dataset_for_evaluation(
                dataset_type, 
                tokenizer, 
                max_samples=args.max_eval_samples
            )
            
            if not eval_data:
                logger.warning(f"Skipping {dataset_type} dataset due to preparation failure.")
                continue
            
            # Special case for TruthfulQA
            if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
                logger.info("Using TruthfulQA binary choice evaluation...")
                from utils.evaluation_utils import evaluate_truthfulness_with_truthfulqa
                
                truthfulqa_metrics = evaluate_truthfulness_with_truthfulqa(
                    model=model,
                    tokenizer=tokenizer,
                    max_length=args.max_length,
                    num_samples=args.max_eval_samples or 100,
                    eval_mode="binary_choice",
                    device=model.device.type
                )
                
                # Add to summary
                summary[dataset_type] = truthfulqa_metrics
                
                # Save detailed results if available
                if "results" in truthfulqa_metrics:
                    dataset_output_dir = os.path.join(output_dir, dataset_type)
                    os.makedirs(dataset_output_dir, exist_ok=True)
                    
                    results_file = os.path.join(dataset_output_dir, "binary_choice_results.json")
                    with open(results_file, "w") as f:
                        json.dump(truthfulqa_metrics["results"], f, indent=2)
                    
                    logger.info(f"TruthfulQA binary choice accuracy: {truthfulqa_metrics.get('truthfulqa_binary_accuracy', 0):.4f}")
                
                continue  # Skip standard evaluation for TruthfulQA
            
            # Generate predictions
            predictions = generate_predictions(
                model, 
                tokenizer, 
                eval_data["prompts"]
            )
            
            # Evaluate hallucination
            dataset_output_dir = os.path.join(output_dir, dataset_type)
            metrics = evaluate_hallucination(
                predictions, 
                eval_data["reference_answers"],
                dataset_type,
                dataset_output_dir
            )
            
            # Add to summary
            summary[dataset_type] = metrics
            
        except Exception as e:
            logger.error(f"Error evaluating on {dataset_type} dataset: {e}")
            logger.error("Skipping this dataset.")
    
    # Save overall summary
    logger.info("Saving evaluation summary...")
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Calculate and save aggregate metrics
    if summary:
        aggregate_metrics = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu1": 0.0,
            "bleu4": 0.0,
            "hallucination_score": 0.0,
        }
        
        for dataset_type, metrics in summary.items():
            for metric in aggregate_metrics:
                aggregate_metrics[metric] += metrics.get(metric, 0.0)
        
        # Average across datasets
        for metric in aggregate_metrics:
            aggregate_metrics[metric] /= len(summary)
        
        logger.info("Aggregate metrics across all datasets:")
        for metric, value in aggregate_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save aggregate metrics
        aggregate_file = os.path.join(output_dir, "aggregate_metrics.json")
        with open(aggregate_file, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 