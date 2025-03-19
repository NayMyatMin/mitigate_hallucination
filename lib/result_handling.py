"""
Functions for saving and processing evaluation results.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def json_encoder(obj):
    """
    Custom JSON encoder for handling special types (numpy, pandas, NaN, etc.).
    
    Args:
        obj: Object to encode
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj):
            return "NaN"
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, tuple):
        return "_".join(str(x) for x in obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif pd.isna(obj):
        return "NaN"
    return obj


def save_metrics_file(metrics: Dict[str, Any], output_dir: str, filename: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save file
        filename: Name of metrics file
    """
    metrics_file = os.path.join(output_dir, filename)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, default=json_encoder, indent=2)
    logger.debug(f"Saved metrics to {metrics_file}")


def save_predictions_to_file(
    dataset_type: str,
    questions: List[str],
    contexts: List[str],
    references: List[str],
    predictions: List[str],
    cleaned_predictions: List[str],
    eval_data: Dict[str, Any],
    dataset_output_dir: str
):
    """
    Save predictions and references to files.
    
    Args:
        dataset_type: Type of dataset
        questions: List of questions
        contexts: List of contexts
        references: List of reference answers
        predictions: List of raw model predictions
        cleaned_predictions: List of cleaned model predictions
        eval_data: Evaluation data dictionary
        dataset_output_dir: Directory to save files
    """
    # Special case for TruthfulQA binary choice
    if dataset_type == "truthfulqa" and eval_data.get("is_binary_choice", False):
        binary_results = []
        for i in range(len(questions)):
            binary_results.append({
                "question": questions[i],
                "choices": eval_data.get("binary_choices", [[]])[i] if i < len(eval_data.get("binary_choices", [])) else [],
                "correct_label": eval_data.get("binary_labels", [[]])[i].index(1) if i < len(eval_data.get("binary_labels", [])) else -1,
                "raw_prediction": predictions[i] if i < len(predictions) else "",
                "cleaned_prediction": cleaned_predictions[i] if i < len(cleaned_predictions) else ""
            })
        
        binary_file = os.path.join(dataset_output_dir, "binary_predictions.json")
        with open(binary_file, "w") as f:
            json.dump(binary_results, f, default=json_encoder, indent=2)
    
    # Special case for HaluEval
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


def save_gpt4o_mini_results(
    evaluation_results: List[Dict[str, Any]], 
    dataset_output_dir: str
):
    """
    Save GPT-4o-mini evaluation results.
    
    Args:
        evaluation_results: List of evaluation results
        dataset_output_dir: Directory to save file
    """
    gpt4o_results_file = os.path.join(dataset_output_dir, "gpt4o_mini_evaluation.json")
    with open(gpt4o_results_file, "w") as f:
        json.dump(evaluation_results, f, default=json_encoder, indent=2)
    logger.debug(f"Saved GPT-4o-mini evaluation results to {gpt4o_results_file}")


def save_analysis_results(
    analysis_results: Dict[str, Any],
    dataset_output_dir: str
):
    """
    Save hallucination analysis results.
    
    Args:
        analysis_results: Dictionary of analysis results
        dataset_output_dir: Directory to save file
    """
    # Save analysis results to JSON
    analysis_file = os.path.join(dataset_output_dir, "hallucination_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis_results, f, default=json_encoder, indent=2)
    
    # Save detailed metrics to CSV if available
    if "individual_metrics" in analysis_results:
        df = pd.DataFrame(analysis_results["individual_metrics"])
        df_file = os.path.join(dataset_output_dir, "hallucination_details.csv")
        df.to_csv(df_file, index=False)
    
    logger.info(f"Hallucination analysis saved to {dataset_output_dir}") 