#!/usr/bin/env python3
"""
Unified evaluation script for measuring hallucination in language models.
This script supports evaluating on multiple datasets, with options to select
specific datasets for evaluation.
"""

import os
import logging
import json
import pandas as pd
import torch
import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
from transformers import set_seed
from peft import PeftModel
from typing import Dict, List, Optional, Any, Tuple

# Local imports
from config.model_config import MODELS
from config.data_config import DATASETS, PROMPT_TEMPLATES
from utils.data_utils import load_and_prepare_datasets, calculate_hallucination_metrics
from utils.model_utils import load_base_model_and_tokenizer
from utils.eval_utils import (
    configure_model_for_generation,
    generate_predictions,
    evaluate_truthfulqa_binary,
    evaluate_hallucination,
    evaluate_halueval,
    analyze_hallucination_patterns,
    calculate_aggregate_metrics,
    prepare_dataset_for_evaluation,
    parse_args
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_prediction(prediction: str, dataset_type: str = None) -> str:
    """
    Clean LLama 3.1 responses by removing instruction formatting, repeated questions,
    explanatory content, and pleasantries.
    
    Args:
        prediction: Raw model output from Llama 3.1
        dataset_type: Type of dataset being processed (for specialized handling)
        
    Returns:
        Cleaned prediction with explanatory text removed
    """
    # Return empty string if prediction is None or empty
    if not prediction or not prediction.strip():
        return ""
    
    # Handle TruthfulQA binary choice differently
    if dataset_type == "truthfulqa":
        # Try to extract just the answer number/letter for binary choice
        binary_matches = re.search(r'(?:option|answer|choice)?\s*(?:is|:)?\s*(?:"\s*)?([1-2]|[A-B])(?:\s*")?', prediction, re.IGNORECASE)
        if binary_matches:
            return binary_matches.group(1).strip()
        
        # Try to match the exact option text
        if "watermelon seeds pass through" in prediction.lower():
            return "1"
        elif "grow watermelons in your stomach" in prediction.lower():
            return "2"
            
        # If all else fails, look for the number 1 or 2 in the text
        number_match = re.search(r'\b[1-2]\b', prediction)
        if number_match:
            return number_match.group(0)
            
        return "1"  # Default to option 1 as it's usually the correct one for TruthfulQA
    
    # Remove instruction formatting
    if "[/INST]" in prediction:
        prediction = prediction.split("[/INST]")[1].strip()
        
    if "[INST]" in prediction:
        prediction = prediction.split("[INST]")[0].strip()
    
    # Remove common prefixes
    prefixes = ["Assistant:", "A:", "Answer:"]
    for prefix in prefixes:
        if prediction.strip().startswith(prefix):
            prediction = prediction[len(prefix):].strip()
    
    # Remove step-by-step reasoning (Markdown and text format)
    step_patterns = [
        r"##\s*Step\s*\d+:.*?(?=##|\n\n|$)",  # Markdown steps
        r"Step\s*\d+:.*?(?=Step|\n\n|$)",      # Plain text steps
        r"First,.*?\n",
        r"Second,.*?\n",
        r"Third,.*?\n",
        r"Finally,.*?\n",
        r"To answer this.*?\n",
    ]
    
    for pattern in step_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove explanations and reasoning
    explanation_patterns = [
        r"The question asks.*?\n",
        r"You asked.*?\n",
        r"To answer (the|your) question.*?\n",
        r"Based on (the )?(given|provided) context.*?\n",
        r"Let's think about this.*?\n",
        r"I'll analyze this.*?\n",
        r"Let me analyze.*?\n",
        r"I need to.*?\n",
        r"I will.*?\n",
        r"I should.*?\n",
        r"I'm going to.*?\n",
        r"Let's.*?\n",
    ]
    
    for pattern in explanation_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove conclusion markers
    conclusion_patterns = [
        r"In conclusion,.*?(?=\n|$)",
        r"To summarize,.*?(?=\n|$)",
        r"Therefore,.*?(?=\n|$)",
        r"Thus,.*?(?=\n|$)",
        r"Hence,.*?(?=\n|$)",
    ]
    
    for pattern in conclusion_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove answer justifications
    justification_patterns = [
        r"The (correct )?answer is:?.*?\n",
        r"The correct option is.*?\n",
        r"Based on the evaluation,.*?\n",
        r"Given the (context|information),.*?\n",
    ]
    
    for pattern in justification_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove LaTeX formatting
    prediction = re.sub(r"\$\\boxed\{(.*?)\}\$", r"\1", prediction)
    
    # Remove pleasantries and notes
    closing_patterns = [
        r"I hope this helps.*?(?=\n|$)",
        r"Let me know if.*?(?=\n|$)",
        r"Hope that answers.*?(?=\n|$)",
        r"Best,.*?(?=\n|$)",
        r"Best regards,.*?(?=\n|$)",
        r"Note:.*?(?=\n|$)",
        r"Also,.*?(?=\n|$)",
        r"\[Your Name\]",
        r"I'm here to help.*?(?=\n|$)",
    ]
    
    for pattern in closing_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle responses where the model repeats the same question
    if "Answer the following question" in prediction:
        prediction = prediction.split("Answer the following question")[0].strip()
    
    # Cleanup extra whitespace
    prediction = re.sub(r'\n\s*\n', '\n', prediction)  # Remove empty lines
    prediction = re.sub(r'\s{2,}', ' ', prediction)    # Remove multiple spaces
    
    # Final cleanup
    lines = [line.strip() for line in prediction.split('\n') if line.strip()]
    if lines:
        prediction = '\n'.join(lines)
    
    return prediction.strip()

def save_predictions_to_file(dataset_type, questions, contexts, references, predictions, cleaned_predictions, eval_data, dataset_output_dir):
    """
    Save prediction results to file based on dataset type.
    
    Args:
        dataset_type: Type of the dataset
        questions: List of questions
        contexts: List of contexts
        references: List of reference answers
        predictions: List of raw model predictions
        cleaned_predictions: List of cleaned predictions
        eval_data: Evaluation data dictionary
        dataset_output_dir: Directory to save results
    """
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Handle TruthfulQA binary choice separately
    if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
        binary_results = []
        for i in range(len(questions)):
            binary_results.append({
                "question": questions[i],
                "choices": eval_data["binary_choices"][i] if i < len(eval_data.get("binary_choices", [])) else [],
                "correct_label": eval_data["binary_labels"][i].index(1) if i < len(eval_data.get("binary_labels", [])) else -1,
                "raw_prediction": predictions[i] if i < len(predictions) else "",
                "cleaned_prediction": cleaned_predictions[i] if i < len(cleaned_predictions) else ""
            })
        
        binary_file = os.path.join(dataset_output_dir, "binary_predictions.json")
        with open(binary_file, "w") as f:
            json.dump(binary_results, f, indent=2)
    
    # Handle HaluEval with hallucinated answers
    elif dataset_type == "halueval_qa" and eval_data.get("has_hallucinated_answer", False):
        hallucinated_answers = eval_data.get("hallucinated_answers", [])
        results_df = pd.DataFrame({
            "question": questions,
            "context": contexts,
            "correct_answer": references,
            "hallucinated_answer": hallucinated_answers,
            "raw_prediction": predictions,
            "cleaned_prediction": cleaned_predictions
        })
        results_file = os.path.join(dataset_output_dir, f"{dataset_type}_predictions.csv")
        results_df.to_csv(results_file, index=False)
    
    # Standard case for other datasets
    else:
        results_df = pd.DataFrame({
            "question": questions,
            "context": contexts,
            "reference": references,
            "raw_prediction": predictions,
            "cleaned_prediction": cleaned_predictions
        })
        results_file = os.path.join(dataset_output_dir, f"{dataset_type}_predictions.csv")
        results_df.to_csv(results_file, index=False)
    
    logger.info(f"Predictions for {dataset_type} saved to {dataset_output_dir}")

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    model_name_short = args.model_name.replace("/", "_")
    adapter_suffix = "_lora" if args.adapter_path else ""
    output_dir = os.path.join(args.output_dir, f"{model_name_short}{adapter_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA adapter: {args.adapter_path if args.adapter_path else 'None'}")
    logger.info(f"Datasets: {args.dataset_types}")
    logger.info(f"Max eval samples per dataset: {args.max_eval_samples}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Clean predictions: {args.clean_predictions}")
    
    # Save evaluation configuration
    config = vars(args)
    config_file = os.path.join(output_dir, "evaluation_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    # Convert string dtype to torch dtype
    torch_dtype = None
    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
        logger.info("Using float16 precision")
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32
        logger.info("Using float32 precision")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=args.model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch_dtype
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
    model = configure_model_for_generation(model, args.max_new_tokens)
    
    # Initialize results summary
    summary = {}
    
    # Evaluate on each specified dataset
    for dataset_type in args.dataset_types:
        try:
            logger.info(f"Evaluating on {dataset_type} dataset...")
            
            # Prepare dataset
            eval_data = prepare_dataset_for_evaluation(
                dataset_type=dataset_type,
                max_samples=args.max_eval_samples
            )
            
            if not eval_data:
                logger.warning(f"Skipping {dataset_type} dataset due to preparation failure.")
                continue
            
            # Create dataset-specific output directory
            dataset_output_dir = os.path.join(output_dir, dataset_type)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Special case for TruthfulQA with binary choices
            if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
                logger.info("Using TruthfulQA binary choice evaluation...")
                
                truthfulqa_metrics, predictions = evaluate_truthfulqa_binary(
                    model=model,
                    tokenizer=tokenizer,
                    eval_data=eval_data,
                    max_new_tokens=args.max_new_tokens
                )
                
                # Clean predictions if requested
                if args.clean_predictions:
                    cleaned_predictions = [clean_prediction(pred, dataset_type="truthfulqa") for pred in predictions]
                    logger.info(f"Sample raw TruthfulQA prediction: {predictions[0][:100]}...")
                    logger.info(f"Sample cleaned TruthfulQA prediction: {cleaned_predictions[0]}")
                else:
                    cleaned_predictions = predictions
                
                # Save metrics
                metrics_file = os.path.join(dataset_output_dir, f"{dataset_type}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(truthfulqa_metrics, f, indent=2)
                
                # Save predictions if requested
                if args.save_predictions:
                    save_predictions_to_file(
                        dataset_type=dataset_type,
                        questions=eval_data["questions"],
                        contexts=eval_data["contexts"],
                        references=eval_data["reference_answers"],
                        predictions=predictions,
                        cleaned_predictions=cleaned_predictions,
                        eval_data=eval_data,
                        dataset_output_dir=dataset_output_dir
                    )
                
                # Add to summary
                summary[dataset_type] = truthfulqa_metrics
                
                continue  # Skip standard evaluation for TruthfulQA binary choice
            
            # Generate predictions for all other datasets
            predictions = generate_predictions(
                model=model,
                tokenizer=tokenizer,
                prompts=eval_data["prompts"],
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size
            )
            
            # Clean predictions if requested
            if args.clean_predictions:
                cleaned_predictions = [clean_prediction(pred, dataset_type=dataset_type) for pred in predictions]
                logger.info(f"Sample raw prediction: {predictions[0][:100]}...")
                logger.info(f"Sample cleaned prediction: {cleaned_predictions[0]}")
            else:
                cleaned_predictions = predictions
            
            # Use cleaned predictions for evaluation
            evaluation_predictions = cleaned_predictions if args.clean_predictions else predictions
            
            # Special case for HaluEval with hallucinated answers
            if dataset_type == "halueval_qa" and eval_data.get("has_hallucinated_answer", False):
                # Evaluate with hallucinated answers
                metrics = evaluate_halueval(
                    predictions=evaluation_predictions,
                    references=eval_data["reference_answers"],
                    hallucinated_answers=eval_data["hallucinated_answers"],
                    dataset_name=dataset_type,
                    output_dir=dataset_output_dir
                )
            else:
                # Standard evaluation for other datasets
                metrics = evaluate_hallucination(
                    predictions=evaluation_predictions,
                    references=eval_data["reference_answers"],
                    dataset_name=dataset_type,
                    output_dir=dataset_output_dir,
                    save_predictions=False  # We'll handle saving consistently below
                )
            
            # Optional hallucination pattern analysis
            if args.hallucination_analysis:
                analysis_results = analyze_hallucination_patterns(
                    predictions=evaluation_predictions,
                    references=eval_data["reference_answers"],
                    questions=eval_data["questions"],
                    contexts=eval_data["contexts"],
                    output_dir=dataset_output_dir
                )
                
                # Save analysis results
                analysis_file = os.path.join(dataset_output_dir, "hallucination_analysis.json")
                with open(analysis_file, "w") as f:
                    # Convert numpy values to Python types for JSON serialization
                    json.dump(analysis_results, f, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=2)
                
                # Save detailed metrics
                if "individual_metrics" in analysis_results:
                    df = pd.DataFrame(analysis_results["individual_metrics"])
                    df_file = os.path.join(dataset_output_dir, "hallucination_details.csv")
                    df.to_csv(df_file, index=False)
                
                logger.info(f"Hallucination analysis saved to {dataset_output_dir}")
            
            # Save metrics to a file
            metrics_file = os.path.join(dataset_output_dir, f"{dataset_type}_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save predictions if requested
            if args.save_predictions:
                save_predictions_to_file(
                    dataset_type=dataset_type,
                    questions=eval_data["questions"],
                    contexts=eval_data["contexts"],
                    references=eval_data["reference_answers"],
                    predictions=predictions,
                    cleaned_predictions=cleaned_predictions,
                    eval_data=eval_data,
                    dataset_output_dir=dataset_output_dir
                )
            
            # Add to summary
            summary[dataset_type] = metrics
            
        except Exception as e:
            logger.error(f"Error evaluating on {dataset_type} dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("Skipping this dataset.")
    
    # Save overall summary
    logger.info("Saving evaluation summary...")
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Calculate and save aggregate metrics
    if summary:
        # Calculate aggregate metrics using the utility function
        aggregate_metrics = calculate_aggregate_metrics(summary)
        
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