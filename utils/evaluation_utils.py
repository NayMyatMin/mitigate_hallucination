"""
Utility functions for model evaluation, focusing on hallucination metrics.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from config.data_config import EVAL_DATASETS


def calculate_hallucination_rate(
    references: List[str], 
    predictions: List[str],
    factual_keywords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate the hallucination rate based on factual keywords.
    
    Args:
        references: List of reference answers
        predictions: List of model predictions
        factual_keywords: List of factual keywords to check for
        
    Returns:
        Dictionary with hallucination metrics
    """
    # If no factual keywords provided, extract from references
    if factual_keywords is None:
        factual_keywords = []
        for ref in references:
            # Extract key entities/facts from reference
            # This is a simplified approach - in production you'd use NER or other techniques
            words = re.findall(r'\b[A-Za-z]{4,}\b', ref)
            factual_keywords.extend([w.lower() for w in words if len(w) > 4])
        
        # Get unique keywords
        factual_keywords = list(set(factual_keywords))
    
    # Calculate metrics
    keyword_presence = []
    for pred in predictions:
        pred_lower = pred.lower()
        # Check what percentage of factual keywords are in the prediction
        if factual_keywords:
            present = sum(1 for kw in factual_keywords if kw.lower() in pred_lower)
            keyword_presence.append(present / len(factual_keywords))
        else:
            keyword_presence.append(1.0)  # No keywords to check
    
    # Calculate metrics
    avg_keyword_presence = np.mean(keyword_presence)
    hallucination_score = 1.0 - avg_keyword_presence
    
    return {
        "hallucination_rate": hallucination_score,
        "factual_consistency": avg_keyword_presence,
    }


def check_citation_quality(
    predictions: List[str],
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Evaluate citation quality in generated text.
    
    Args:
        predictions: List of model predictions
        sources: List of dictionaries with source information
        
    Returns:
        Dictionary with citation quality metrics
    """
    # Check for citation patterns in predictions
    citation_pattern = r'(\[[\d,\s]+\]|\.[\d]+|Citation:[\s\d,]+)'
    
    has_citation = [1 if re.search(citation_pattern, pred) else 0 for pred in predictions]
    citation_rate = sum(has_citation) / len(predictions)
    
    # If sources are provided, check if citations match sources
    citation_accuracy = 0.0
    if sources:
        # This would need to be implemented based on your citation format
        # For now, we'll just return the citation rate
        citation_accuracy = citation_rate
    
    return {
        "citation_rate": citation_rate,
        "citation_accuracy": citation_accuracy,
    }


def evaluate_truthfulness_with_truthfulqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate model truthfulness using TruthfulQA benchmark.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        max_length: Maximum generation length
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with TruthfulQA metrics
    """
    # Load TruthfulQA dataset
    try:
        dataset = load_dataset("truthful_qa", "multiple_choice")
        if "validation" in dataset:
            dataset = dataset["validation"]
        else:
            dataset = dataset["test"]
        
        # Limit to num_samples
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        # Format questions for the model
        questions = [f"[INST] {q} [/INST]" for q in dataset["question"]]
        
        # Generate answers
        model_outputs = []
        for question in questions:
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_return_sequences=1,
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the model's response after the instruction
            response = output_text.split("[/INST]")[-1].strip()
            model_outputs.append(response)
        
        # Analyze truthfulness
        # This is a simplified approach - actual TruthfulQA evaluation is more complex
        correct_count = 0
        for i, output in enumerate(model_outputs):
            # For simplicity, we check if the model's answer contains any correct answers
            correct_answers = dataset["correct_answers"][i]
            output_lower = output.lower()
            for ans in correct_answers:
                if ans.lower() in output_lower:
                    correct_count += 1
                    break
        
        truthfulness_score = correct_count / len(model_outputs)
        return {
            "truthfulqa_score": truthfulness_score,
        }
    
    except Exception as e:
        print(f"Error evaluating with TruthfulQA: {e}")
        return {
            "truthfulqa_score": 0.0,
        }


def evaluate_hallucination(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    references_key: str = "output",
    input_key: str = "input",
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of hallucination in model outputs.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset for evaluation
        references_key: Key for reference answers in dataset
        input_key: Key for input questions in dataset
        max_length: Maximum generation length
        
    Returns:
        Dictionary with hallucination metrics
    """
    if len(eval_dataset) == 0:
        return {
            "hallucination_rate": 0.0,
            "factual_consistency": 0.0,
            "citation_rate": 0.0,
        }
    
    # Format inputs
    inputs = [f"[INST] {example} [/INST]" for example in eval_dataset[input_key]]
    references = eval_dataset[references_key]
    
    # Generate predictions
    predictions = []
    for input_text in inputs:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response after the instruction
        response = output_text.split("[/INST]")[-1].strip()
        predictions.append(response)
    
    # Calculate hallucination metrics
    hallucination_metrics = calculate_hallucination_rate(references, predictions)
    
    # Calculate citation metrics
    citation_metrics = check_citation_quality(predictions)
    
    # Combine metrics
    combined_metrics = {**hallucination_metrics, **citation_metrics}
    
    return combined_metrics 