#!/usr/bin/env python3
"""
Evaluation script for hallucination mitigation models.
"""

import os
import argparse
import logging
import json
from typing import Dict, Any, List, Optional
import torch
from datasets import load_dataset

from config.model_config import MODELS
from config.data_config import EVAL_DATASETS
from utils.model_utils import load_base_model_and_tokenizer, load_finetuned_model
from utils.evaluation_utils import (
    evaluate_hallucination,
    evaluate_truthfulness_with_truthfulqa
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a model on hallucination metrics")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        help="Name of the base model (from config)",
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter. If not provided, will evaluate the base model.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
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
        "--eval_truthfulqa",
        action="store_true",
        help="Whether to evaluate on TruthfulQA",
    )
    
    parser.add_argument(
        "--eval_custom",
        action="store_true",
        help="Whether to evaluate on custom hallucination datasets",
    )
    
    parser.add_argument(
        "--custom_dataset_path",
        type=str,
        default=None,
        help="Path to custom evaluation dataset",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate",
    )
    
    return parser.parse_args()


def evaluate_model(
    model,
    tokenizer,
    args,
) -> Dict[str, Any]:
    """
    Evaluate model on hallucination metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Evaluate on TruthfulQA if requested
    if args.eval_truthfulqa:
        logger.info("Evaluating on TruthfulQA...")
        truthfulqa_results = evaluate_truthfulness_with_truthfulqa(
            model=model,
            tokenizer=tokenizer,
            num_samples=args.max_samples
        )
        results.update(truthfulqa_results)
    
    # Evaluate on custom datasets if requested
    if args.eval_custom:
        if args.custom_dataset_path:
            # Load custom dataset
            logger.info(f"Loading custom dataset from {args.custom_dataset_path}")
            
            try:
                if os.path.isfile(args.custom_dataset_path):
                    # Load from local file
                    dataset = load_dataset(
                        "json",
                        data_files=args.custom_dataset_path,
                        split="train"
                    )
                else:
                    # Load from Hugging Face Hub
                    dataset = load_dataset(args.custom_dataset_path)
                    if "validation" in dataset:
                        dataset = dataset["validation"]
                    else:
                        dataset = dataset["test"]
                
                # Limit to max_samples
                if len(dataset) > args.max_samples:
                    dataset = dataset.select(range(args.max_samples))
                
                # Evaluate on custom dataset
                logger.info("Evaluating on custom dataset...")
                custom_results = evaluate_hallucination(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataset=dataset,
                    # Assume default keys, can be customized if needed
                    input_key="input",
                    references_key="output"
                )
                
                results.update({
                    "custom_" + k: v for k, v in custom_results.items()
                })
                
            except Exception as e:
                logger.error(f"Error evaluating on custom dataset: {e}")
        else:
            logger.warning("--eval_custom specified but no custom_dataset_path provided.")
    
    return results


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model type for results file name
    model_type = "base"
    if args.adapter_path:
        # Extract adapter name from path
        adapter_name = os.path.basename(os.path.normpath(args.adapter_path))
        model_type = f"lora_{adapter_name}"
    
    # Set up output path
    results_path = os.path.join(
        args.output_dir,
        f"{args.model_name}_{model_type}_results.json"
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    if args.adapter_path:
        # Load fine-tuned model with adapter
        model, tokenizer = load_finetuned_model(
            base_model_name=args.model_name,
            adapter_path=args.adapter_path,
            load_in_8bit=args.load_in_8bit,
        )
    else:
        # Load base model
        model, tokenizer = load_base_model_and_tokenizer(
            model_name=args.model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
    
    # Make sure model is in evaluation mode
    model.eval()
    
    # Log model information
    logger.info(f"Model: {args.model_name}")
    if args.adapter_path:
        logger.info(f"Adapter: {args.adapter_path}")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(model, tokenizer, args)
    
    # Log evaluation results
    logger.info(f"Evaluation results: {results}")
    
    # Save evaluation results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")


if __name__ == "__main__":
    main() 