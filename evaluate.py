#!/usr/bin/env python3
"""
Unified evaluation script for measuring hallucination in language models.
This script supports evaluating on multiple datasets, with options to select
specific datasets for evaluation.
"""

import os
import logging
from transformers import set_seed
from utils.eval_utils import parse_args, calculate_aggregate_metrics

# Import our library modules
from lib.config_handling import (
    setup_logging,
    setup_output_directory,
    save_config,
    setup_model,
    log_evaluation_start
)
from lib.evaluation import evaluate_dataset
from lib.result_handling import save_metrics_file

# Set up logger
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Create output directory
    output_dir = setup_output_directory(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        base_dir=args.output_dir
    )
    
    # Log configuration
    log_evaluation_start(args)
    
    # Save evaluation configuration
    save_config(vars(args), output_dir)
    
    # Set up model and tokenizer
    model, tokenizer = setup_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens
    )
    
    # Initialize results summary
    summary = {}
    
    # Evaluate on each specified dataset
    for dataset_type in args.dataset_types:
        # Evaluate dataset
        metrics = evaluate_dataset(
            dataset_type=dataset_type,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            max_eval_samples=args.max_eval_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
            clean_predictions=args.clean_predictions,
            save_predictions=args.save_predictions,
            hallucination_analysis=args.hallucination_analysis,
            use_gpt4o_mini=args.use_gpt4o_mini if hasattr(args, 'use_gpt4o_mini') else False,
            openai_api_key=args.openai_api_key if hasattr(args, 'openai_api_key') else None
        )
        
        # Add results to summary
        if metrics:
            summary[dataset_type] = metrics
    
    # Save overall summary
    logger.info("Saving evaluation summary...")
    save_metrics_file(summary, output_dir, "evaluation_summary.json")
    
    # Calculate and save aggregate metrics
    if summary:
        # Calculate aggregate metrics using the utility function
        aggregate_metrics = calculate_aggregate_metrics(summary)
        
        logger.info("Aggregate metrics across all datasets:")
        for metric, value in aggregate_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save aggregate metrics
        save_metrics_file(aggregate_metrics, output_dir, "aggregate_metrics.json")
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 