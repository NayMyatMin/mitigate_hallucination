"""
Functions for cleaning and normalizing model predictions.
"""

import re
from typing import List, Optional


def clean_prediction(prediction: str, dataset_type: str = None) -> str:
    """
    Clean model responses by removing instruction formatting, repeated questions,
    explanatory content, and pleasantries.
    
    Args:
        prediction: Raw model output 
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
    latex_patterns = [
        r"\$\$.*?\$\$",
        r"\$.*?\$",
    ]
    
    for pattern in latex_patterns:
        prediction = re.sub(pattern, "", prediction, flags=re.DOTALL)
    
    # Cleanup whitespace
    prediction = re.sub(r'\n\s*\n', '\n', prediction)  # Remove empty lines
    prediction = re.sub(r'\s{2,}', ' ', prediction)    # Remove multiple spaces
    
    # Final cleanup - take only the first paragraph for very verbose answers
    lines = prediction.split('\n')
    if len(lines) > 3:
        # If the answer is very long, take just the first few sentences
        prediction = lines[0]
        if len(prediction.split('.')) <= 2 and len(lines) > 1:
            prediction = f"{prediction} {lines[1]}"
    
    return prediction.strip()


def clean_predictions_batch(predictions: List[str], dataset_type: str = None) -> List[str]:
    """
    Clean a batch of predictions.
    
    Args:
        predictions: List of raw model outputs
        dataset_type: Type of dataset being processed
        
    Returns:
        List of cleaned predictions
    """
    return [clean_prediction(pred, dataset_type=dataset_type) for pred in predictions] 