"""
Core evaluation functions for processing datasets and generating metrics.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.eval_utils import (
    prepare_dataset_for_evaluation,
    generate_predictions,
    evaluate_truthfulqa_binary,
    evaluate_hallucination,
    evaluate_halueval,
    analyze_hallucination_patterns,
    evaluate_with_gpt4o_mini,
    calculate_gpt4o_mini_metrics
)

from lib.prediction_cleaning import clean_predictions_batch, clean_prediction
from lib.result_handling import (
    save_metrics_file,
    save_predictions_to_file,
    save_gpt4o_mini_results,
    save_analysis_results
)

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset_type: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    max_eval_samples: int = 100,
    max_new_tokens: int = 256,
    batch_size: int = 1,
    clean_predictions: bool = True,
    save_predictions: bool = True,
    hallucination_analysis: bool = True,
    use_gpt4o_mini: bool = False,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single dataset for hallucination.
    
    Args:
        dataset_type: Type of dataset to evaluate
        model: Model to use for evaluation
        tokenizer: Tokenizer for the model
        output_dir: Base output directory
        max_eval_samples: Maximum number of samples to evaluate
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation
        clean_predictions: Whether to clean predictions
        save_predictions: Whether to save predictions
        hallucination_analysis: Whether to perform hallucination analysis
        use_gpt4o_mini: Whether to use GPT-4o-mini for evaluation
        openai_api_key: OpenAI API key for GPT-4o-mini
        
    Returns:
        Metrics dictionary
    """
    try:
        logger.info(f"Evaluating on {dataset_type} dataset...")
        
        # Prepare dataset
        eval_data = prepare_dataset_for_evaluation(
            dataset_type=dataset_type,
            max_samples=max_eval_samples
        )
        
        if not eval_data:
            logger.warning(f"Skipping {dataset_type} dataset due to preparation failure.")
            return {}
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(output_dir, dataset_type)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Special case for TruthfulQA with binary choices
        if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
            return evaluate_truthfulqa_dataset(
                model=model,
                tokenizer=tokenizer,
                eval_data=eval_data,
                dataset_output_dir=dataset_output_dir,
                max_new_tokens=max_new_tokens,
                clean_predictions=clean_predictions,
                save_predictions=save_predictions,
                use_gpt4o_mini=use_gpt4o_mini,
                openai_api_key=openai_api_key
            )
        
        # Generate predictions
        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            prompts=eval_data["prompts"],
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )
        
        # Clean predictions if requested
        if clean_predictions:
            cleaned_predictions = clean_predictions_batch(predictions, dataset_type=dataset_type)
            logger.info(f"Sample raw prediction: {predictions[0][:100]}...")
            logger.info(f"Sample cleaned prediction: {cleaned_predictions[0]}")
        else:
            cleaned_predictions = predictions
        
        # Use cleaned predictions for evaluation
        evaluation_predictions = cleaned_predictions if clean_predictions else predictions
        
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
        
        # Add GPT-4o-mini evaluation if requested
        if use_gpt4o_mini:
            logger.info(f"Evaluating {dataset_type} with GPT-4o-mini...")
            
            gpt4o_mini_results = evaluate_with_gpt4o_mini(
                questions=eval_data["questions"],
                contexts=eval_data["contexts"],
                references=eval_data["reference_answers"],
                predictions=evaluation_predictions,
                openai_api_key=openai_api_key,
                dataset_type=dataset_type
            )
            
            if gpt4o_mini_results:
                # Calculate GPT-4o-mini metrics
                gpt4o_mini_metrics = calculate_gpt4o_mini_metrics(gpt4o_mini_results)
                
                # Add GPT-4o-mini metrics to regular metrics
                metrics.update(gpt4o_mini_metrics)
                
                # Save GPT-4o-mini evaluation results
                save_gpt4o_mini_results(gpt4o_mini_results, dataset_output_dir)
        
        # Optional hallucination pattern analysis
        if hallucination_analysis:
            analysis_results = analyze_hallucination_patterns(
                predictions=evaluation_predictions,
                references=eval_data["reference_answers"],
                questions=eval_data["questions"],
                contexts=eval_data["contexts"],
                output_dir=dataset_output_dir
            )
            
            # Save analysis results
            save_analysis_results(analysis_results, dataset_output_dir)
        
        # Save metrics to a file
        save_metrics_file(metrics, dataset_output_dir, f"{dataset_type}_metrics.json")
        
        # Save predictions if requested
        if save_predictions:
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
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating on {dataset_type} dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Skipping this dataset.")
        return {}


def evaluate_truthfulqa_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: Dict[str, Any],
    dataset_output_dir: str,
    max_new_tokens: int = 256,
    clean_predictions: bool = True,
    save_predictions: bool = True,
    use_gpt4o_mini: bool = False,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate TruthfulQA binary choice dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data dictionary
        dataset_output_dir: Output directory for this dataset
        max_new_tokens: Maximum number of tokens to generate
        clean_predictions: Whether to clean predictions
        save_predictions: Whether to save predictions
        use_gpt4o_mini: Whether to use GPT-4o-mini
        openai_api_key: OpenAI API key
        
    Returns:
        Metrics dictionary
    """
    logger.info("Using TruthfulQA binary choice evaluation...")
    
    truthfulqa_metrics, predictions = evaluate_truthfulqa_binary(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        max_new_tokens=max_new_tokens
    )
    
    # Clean predictions if requested
    if clean_predictions:
        cleaned_predictions = [clean_prediction(pred, dataset_type="truthfulqa") for pred in predictions]
        logger.info(f"Sample raw TruthfulQA prediction: {predictions[0][:100]}...")
        logger.info(f"Sample cleaned TruthfulQA prediction: {cleaned_predictions[0]}")
    else:
        cleaned_predictions = predictions
    
    # Add GPT-4o-mini evaluation if requested
    if use_gpt4o_mini:
        logger.info("Evaluating TruthfulQA with GPT-4o-mini...")
        
        gpt4o_mini_results = evaluate_with_gpt4o_mini(
            questions=eval_data["questions"],
            contexts=["" for _ in eval_data["questions"]],  # TruthfulQA has no context
            references=eval_data["reference_answers"],
            predictions=cleaned_predictions if clean_predictions else predictions,
            openai_api_key=openai_api_key,
            dataset_type="truthfulqa"
        )
        
        if gpt4o_mini_results:
            # Calculate GPT-4o-mini metrics
            gpt4o_mini_metrics = calculate_gpt4o_mini_metrics(gpt4o_mini_results)
            
            # Add GPT-4o-mini metrics to TruthfulQA metrics
            truthfulqa_metrics.update(gpt4o_mini_metrics)
            
            # Save GPT-4o-mini evaluation results
            save_gpt4o_mini_results(gpt4o_mini_results, dataset_output_dir)
    
    # Save metrics to a file
    save_metrics_file(truthfulqa_metrics, dataset_output_dir, "truthfulqa_metrics.json")
    
    # Save predictions if requested
    if save_predictions:
        save_predictions_to_file(
            dataset_type="truthfulqa",
            questions=eval_data["questions"],
            contexts=eval_data["contexts"],
            references=eval_data["reference_answers"],
            predictions=predictions,
            cleaned_predictions=cleaned_predictions,
            eval_data=eval_data,
            dataset_output_dir=dataset_output_dir
        )
    
    return truthfulqa_metrics 